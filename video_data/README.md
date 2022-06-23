### Split long video into small chunks
----

- Folder format:
    ```
    -anatomy_videos
        |-- video_1.mp4
        |-- video_1.txt
        ...
        |-- video_10.mp4
        |-- video_10.txt
    ```


- Execution:

    - ``` -s : source path of video file (ex: .mp4) ```
    - ``` -d : destination path of processed chunk video files (ex: .mp4) ```

    <!-- - ``` -t : path of times.txt file which is included 'start-time' and 'end-time' in seconds. ``` -->

    - Like:
        ```python
        python split_video.py -s <source path of video file (.mp4)> -d <destination path of processed chunk video files (.mp4)> 
        ```
    - For execution of `split_video.py` run following command: 
    ```python
    python split_video.py -s anatomy_videos/ -d split_videos/
    ```
