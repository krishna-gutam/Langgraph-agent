from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool

@tool
def get_youtube_transcript(video_id: str):
    """
    Extracts the transcript for a given YouTube video ID.
    
    Args:
        video_id: The YouTube video ID (e.g., 'dQw4w9WgXcQ').
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# Assign name and description for discovery
get_youtube_transcript.name = "get_youtube_transcript"
get_youtube_transcript.description = "Extracts the transcript for a given YouTube video ID."
