from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS
from urllib.parse import urlparse, parse_qs
from .base import LLMTool
from typing import Any
class YouTubeSearchTool(LLMTool):
    name: str = "YouTube Search Tool"
    description: str = "A tool capable of searching the internet for youtube videos and returns the text transcript of the videos"
    arg: str = "A single string parameter that will be searched on the internet to find relevant content"
    # Specific Parameters
    ddgs: Any = None
    def __init__(self, **data):
        super().__init__(**data)
        if self.ddgs is None:
            self.ddgs = DDGS()
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL."""
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/shorts/'):
                return parsed_url.path.split('/')[2]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        return None
    def search_videos(self, query, max_results=5):
        """Search YouTube videos using DuckDuckGo."""
        try:
            # Search for videos using DDG videos search
            results = self.ddgs.videos(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                timelimit="w",
                resolution="high",
                duration="medium",
                max_results=max_results*2 # Get 2x required results so get some relevant results. Sort and Filter later.
            )
            results = sorted(
                results,
                key=lambda x: (-(x['statistics']['viewCount'] if x['statistics']['viewCount'] is not None else float('-inf'))))[:max_results] # sort by more views --> first
            videos = []
            for result in results:
                video_url = result.get('content')  # The actual video URL is in the 'content' field
                video_id = self.extract_video_id(video_url)
                if video_id:
                    video_data = {
                        'title': result['title'],
                        'video_id': video_id,
                        'description': result.get('description', ''),
                        'link': video_url,
                        'duration': result.get('duration', ''),
                        'publisher': result.get('publisher', ''),
                        'uploader': result.get('uploader', ''),
                        'published': result.get('published', ''),
                        'view_count': result.get('statistics', {}).get('viewCount', 'N/A'),
                        'thumbnail': result.get('images', {}).get('large', '')
                    }
                    videos.append(video_data)
            if not videos:
                return "No YouTube videos found in the search results."
            return videos[:max_results]
        except Exception as e:
            return f"Error searching videos: {str(e)}"
    def get_transcript(self, video_id):
        """Get transcript for a YouTube video."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript_list])
        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            return None
    #def summarize_content(self, transcript):
    #    prompt = "Create a concise summary of the following video transcript"
    #    try:
    #        response = self.client.chat.completions.create(
    #            model="gpt-4",
    #            messages=[
    #                {"role": "system", "content": "You are an expert content creator specializing in creating high-quality content from video transcripts."},
    #                {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
    #            ],
    #            max_tokens=2000)
    #        return response.choices[0].message.content.strip()
    #    except Exception as e:
    #        return None
    def summarize_content(self, transcript):
        prompt = "Create a concise summary of the following video transcript"
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default to gpt-4 if MODEL_NAME is not set
        try:
            if openrouter_api_key:
                print(f"Using OpenRouter with model: {model_name} for content summarization")
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert content creator specializing in creating high-quality content from video transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
                    ],
                    max_tokens=2000)
            else: # Fall back to default OpenAI client
                print("OpenRouter API key not found, using default OpenAI client with gpt-4")
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert content creator specializing in creating high-quality content from video transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
                    ],
                    max_tokens=2000)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with primary model: {e}")
            print("Falling back to default OpenAI client with gpt-4")
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert content creator specializing in creating high-quality content from video transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"}
                    ],
                    max_tokens=2000)
                return response.choices[0].message.content.strip()
            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return None
    def run(self, prompt: str) -> str:
        print(f"Calling YouTube Search Tool with prompt: {prompt}")
        try: # Search for videos
            videos = self.search_videos(prompt, 3)
            if isinstance(videos, str):  # Error occurred
                return f"Search error: {videos}"
            if not videos:  # No videos found
                return "No videos found matching the query."
            results = []
            for video in videos: # Get transcript
                transcript = self.get_transcript(video['video_id'])
                if not transcript:
                    continue
                content = self.summarize_content(transcript)
                results.append({
                    "video": video,
                    "content": content.replace("\n\n", "\n").replace("\n\n\n", "\n")
                 })
            if not results:
                return "Could not process any videos. Try a different search query."
            results = list(map(lambda x: f"Video Title: {x['video']['title']}\nContent: {x['content']}", results))
            return "\n\n\n".join(results)
        except Exception as e:
            return f"Error executing task: {str(e)}"
