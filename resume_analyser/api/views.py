"""
views.py — REST API + HTML page views for Resume Analyser.
"""
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import ResumeUpload
from .serializers import ResumeUploadSerializer, ResumeHistorySerializer, JDMatchSerializer
from . import services

ALLOWED_EXTENSIONS = {'.pdf', '.docx'}


def _save_file(uploaded_file) -> str:
    """Save uploaded file to MEDIA_ROOT and return its path."""
    ext = Path(uploaded_file.name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_dir = Path(settings.MEDIA_ROOT)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / unique_name

    with open(save_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    return str(save_path)


# ─── REST API Views ────────────────────────────────────────────────────────────

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_resume(request):
    """POST /api/upload/ — Upload and store a resume file."""
    if 'file' not in request.FILES:
        return Response(
            {"error": "No file provided. Please upload a PDF or DOCX file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    uploaded_file = request.FILES['file']
    ext = Path(uploaded_file.name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return Response(
            {"error": f"Invalid file format '{ext}'. Only PDF and DOCX files are accepted."},
            status=status.HTTP_400_BAD_REQUEST
        )

    file_path = _save_file(uploaded_file)
    resume = ResumeUpload.objects.create(
        file_name=uploaded_file.name,
        file_path=file_path,
        status='pending',
    )

    return Response({
        "message": "File uploaded successfully.",
        "resume_id": resume.id,
        "file_name": resume.file_name,
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@parser_classes([JSONParser])
def analyze_resume(request):
    """POST /api/analyze/ — Run NLP + ML analysis on an uploaded resume."""
    resume_id = request.data.get('resume_id')
    if not resume_id:
        return Response({"error": "resume_id is required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        resume = ResumeUpload.objects.get(id=resume_id)
    except ResumeUpload.DoesNotExist:
        return Response({"error": "Resume not found."}, status=status.HTTP_404_NOT_FOUND)

    if not os.path.exists(resume.file_path):
        return Response({"error": "Resume file not found on disk."}, status=status.HTTP_404_NOT_FOUND)

    try:
        result = services.analyze_resume(resume.file_path)
    except Exception as e:
        return Response({"error": f"Analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Persist results
    resume.status = result['status']
    resume.category = result.get('category') or ''
    resume.score = result.get('score', 0)
    resume.extracted_text = result.get('extracted_text', '')
    resume.skills_found = result.get('skills_found', [])
    resume.suggestions = result.get('suggestions', [])
    resume.save()

    return Response({
        "resume_id": resume.id,
        "file_name": resume.file_name,
        "status": result['status'],
        "reason": result.get('reason', ''),
        "category": result.get('category'),
        "confidence": result.get('confidence'),
        "score": result.get('score', 0),
        "word_count": result.get('word_count', 0),
        "skills_found": result.get('skills_found', []),
        "suggestions": result.get('suggestions', []),
    })


@api_view(['POST'])
@parser_classes([JSONParser])
def match_jd(request):
    """POST /api/match-jd/ — Compare resume against a job description."""
    serializer = JDMatchSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    resume_id = serializer.validated_data['resume_id']
    jd_text = serializer.validated_data['job_description']

    try:
        resume = ResumeUpload.objects.get(id=resume_id)
    except ResumeUpload.DoesNotExist:
        return Response({"error": "Resume not found."}, status=status.HTTP_404_NOT_FOUND)

    if not resume.extracted_text:
        return Response(
            {"error": "Please run /api/analyze/ first before matching."},
            status=status.HTTP_400_BAD_REQUEST
        )

    result = services.match_job_description(resume.extracted_text, jd_text)

    # Persist JD match results
    resume.jd_match_percent = result['match_percent']
    resume.matching_skills = result['matching_skills']
    resume.missing_skills = result['missing_skills']
    resume.save()

    return Response({
        "resume_id": resume.id,
        "file_name": resume.file_name,
        "match_percent": result['match_percent'],
        "matching_skills": result['matching_skills'],
        "missing_skills": result['missing_skills'],
        "feedback": result['feedback'],
    })


@api_view(['GET'])
def history(request):
    """GET /api/history/ — List all analysed resumes."""
    resumes = ResumeUpload.objects.all()
    serializer = ResumeHistorySerializer(resumes, many=True)
    return Response(serializer.data)


# ─── HTML Page Views ───────────────────────────────────────────────────────────

def home(request):
    return render(request, 'home.html')


def upload_page(request):
    return render(request, 'upload.html')


def result_page(request, resume_id):
    resume = get_object_or_404(ResumeUpload, id=resume_id)
    return render(request, 'result.html', {'resume': resume})


def history_page(request):
    resumes = ResumeUpload.objects.all()
    return render(request, 'history.html', {'resumes': resumes})
