from rest_framework import serializers
from .models import ResumeUpload


class ResumeUploadSerializer(serializers.ModelSerializer):
    """Serializer for the ResumeUpload model."""

    class Meta:
        model = ResumeUpload
        fields = '__all__'
        read_only_fields = [
            'uploaded_at', 'status', 'category', 'score',
            'extracted_text', 'skills_found', 'suggestions',
            'jd_match_percent', 'matching_skills', 'missing_skills'
        ]


class ResumeHistorySerializer(serializers.ModelSerializer):
    """Lightweight serializer for the history list view."""

    class Meta:
        model = ResumeUpload
        fields = [
            'id', 'file_name', 'uploaded_at', 'status',
            'category', 'score', 'jd_match_percent'
        ]


class JDMatchSerializer(serializers.Serializer):
    """Serializer for job description matching request."""
    resume_id = serializers.IntegerField()
    job_description = serializers.CharField(min_length=20)
