### Split long video into small chunks
----

- Execution:

    - ``` -s : source path of video file (.mp4) ```
    - ``` -d : destination path of processed chunk video files (.mp4) ```
    - ``` -t : path of times.txt file which is included 'start-time' and 'end-time' in seconds. ```

    - Like:
        ```python
        python split_video.py -s <source path of video file (.mp4)> -d <destination path of processed chunk video files (.mp4)> -t <path of times.txt file which is included 'start-time' and 'end-time' in seconds.>
        ```
    - For execution of `split_video.py` run following command: 
    ```python
    python split_video.py -s anatomy_videos/General_embryology_Part-1.mp4 -d split_videos/ -t times.txt
    ```
