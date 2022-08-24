import json
import os

import pandas as pd
import requests
from youtube_transcript_api import YouTubeTranscriptApi

from analyzer import analyze
from model import train_random_forrest_regressor, train_random_ada_boost_regressor, train_deep_regression_model

channel_id = "UC1-VOKyTJrgLiBeiJqzeIUQ"
api_key = "Insert Your API Key"

file_less_data = 'features_less_data.csv'
file_more_data_without_outlier_removal = 'features_more_data_without_outlier_removal.csv'
file_more_data_with_outlier_removal = 'features_more_data_with_outlier_removal.csv'

video_ids = []
base_path = "/out"


def get_all_videos_for_channel(channelId, before_date=""):
    next_page_token = ""

    while True:
        request_url = "https://www.googleapis.com/youtube/v3/search?key=" + api_key + "&channelId=" \
                      + channelId + "&part=snippet,id&order=date&pageToken=" + next_page_token

        if before_date != "":
            request_url = request_url + "&publishedBefore=" + before_date

        r = requests.get(request_url)

        response = json.loads(r.text)

        if response.get('nextPageToken') is None:
            break

        next_page_token = response['nextPageToken']

        for entry in response['items']:
            video_ids.append(entry['id']['videoId'])


def get_transcript_for_video(video_id):
    text = ""

    try:
        # retrieve the available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = transcript_list.find_generated_transcript(['de'])

        # fetch the actual transcript data
        transcript_text = transcript.fetch()

        for text_part in transcript_text:
            text = text + "\n" + text_part['text']

        return text
    except:
        print("No transcript for this video: " + video_id)
        return text


def get_statistics_for_video(video_id):
    request_url = "https://www.googleapis.com/youtube/v3/videos?part=statistics&id=" + video_id + "&key=" + api_key

    r = requests.get(request_url)

    response = json.loads(r.text)

    view_count = response['items'][0]['statistics']['viewCount']
    like_count = response['items'][0]['statistics']['likeCount']
    favorite_count = response['items'][0]['statistics']['favoriteCount']
    comment_count = response['items'][0]['statistics']['commentCount']

    return view_count, like_count, favorite_count, comment_count


def crawl_data(before_date="", take_existing_video_ids=True):
    out_directory = 'out/'
    for f in os.listdir(out_directory):
        os.remove(os.path.join(out_directory, f))

    if not take_existing_video_ids:
        get_all_videos_for_channel(channel_id, before_date)
    else:
        df = pd.read_csv("video_ids.csv", header=0, delimiter=",")
        df.drop_duplicates(subset="video_ids",
                           keep=False, inplace=True)
        for v in df[["video_ids"]].values:
            video_ids.append(v[0])

    for video_id in video_ids:
        transcript = get_transcript_for_video(video_id)

        if transcript == "":
            continue

        view_count, like_count, favorite_count, comment_count = get_statistics_for_video(video_id)

        f = open("out/transcript%" + video_id + "%" + view_count + "%" + like_count + "%" + favorite_count
                 + "%" + comment_count + ".txt", "x")
        f.write(transcript)
        f.close()


if __name__ == '__main__':
    crawl_data()

    analyze(translate_new=True)

    train_deep_regression_model(file_less_data)
    train_random_forrest_regressor(file_less_data)
    train_random_ada_boost_regressor(file_less_data)
