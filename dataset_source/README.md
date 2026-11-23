---
license: other
pretty_name: "PS04 Aerial View Image Reasoning (student data)"
tags: [hackathon, aerial]
---
# PS04: Aerial View Image Reasoning

**Theme:** Model Training | 24-hour hackathon, teams of two, AI coding tools allowed.

## What you build
Answer counting and comparison questions about before-and-after satellite images of disaster areas, and say 'cannot be determined' when the images cannot answer.

## What is in this repository
400 training image pairs (1024x1024) from 8 disasters with building outlines and damage labels, 40 practice pairs with 50 questions and answers, and the question templates with the rules.

## Output format
JSONL: `{"id","answer","region"}`, region = `[x1,y1,x2,y2]` or a quadrant/half name.

## Source and licence
xBD / xView2 (Gupta et al. 2019; Maxar Open Data), CC BY-NC-SA 4.0. Images were cropped, rotated and re-encoded by the organizers. Use this data only for the event and for non-commercial work, with attribution. Hidden test data and answer keys are not included.
Read the student problem statement document for the full task, scoring and rules.
