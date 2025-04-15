# Project PU

## Test the stream
In Home, run the python script and check if the stream is transmitting correctly. 

<!-- The script receives an argument in terminal, which is the camera number. -->

    python3 test_cam.py     

## Enter the container

    docker exec -ti briq_container bash

## Download the base YOLO model and export it

Download

    yolo predict model=yolo11n.pt source='https://ultralytics.com/images/zidane.jpg'

Export to engine format

    yolo mode=export model=yolo11n.pt format=engine device=0 half=True

### integrado
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    

## Create YML file

Create the YML file and add the characteristics of all the cameras.

    touch mkdocs.yml
    sudo nano mkdocs.yml

Add this lines:

    site_name: torch2trt
    theme:
        name: "material"
        palette:
            primary: green
            secondary: light green

    repo_url: https://github.com/NVIDIA-AI-IOT/torch2trt

    plugins:
    - search
    
    use_directory_urls: False

    edit_uri: blob/master/docs
    markdown_extensions:
    - pymdownx.tabbed
    - pymdownx.keys
    - pymdownx.snippets
    - pymdownx.inlinehilite
    - pymdownx.highlight:
        use_pygments: true
    - admonition
    - pymdownx.details
    - pymdownx.superfences
    - attr_list
    
    # use_directory_urls - False to fix broken raw html image links
    # https://github.com/mkdocs/mkdocs/issues/991


    nav:

    - Home: index.md
    - Getting Started: getting_started.md
    - Usage:
        - Basic Usage: usage/basic_usage.md
        - Reduced Precision: usage/reduced_precision.md
        - Custom Converter: usage/custom_converter.md
    - Converters: converters.md
    - Benchmarks: 
        - Jetson Nano: benchmarks/jetson_nano.md
        - Jetson Xavier: benchmarks/jetson_xavier.md
    - Contributing: CONTRIBUTING.md
    - See Also: see_also.md

    extra_css:
        - css/version-select.css
    extra_javascript:
        - js/version-select.js
        
    google_analytics:
        - UA-135919510-3
        - auto
        
    # Configuración de cámaras
    cameras:
    1:
        input_video: "rtsp://admin:Perfumeriasunidas2!@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0"
        # input_video: "camera_1_video.mp4"
        output_video: "camera_1_processed.mp4"
        camera_sn: "AD0B109PAZ473AF-1"
        polygons:
        - [1, [[99,9],[123,171],[133,224],[189,172],[167,153],[151,3]]]
        - [2, [[151,3],[197,3],[201,103],[190,133],[167,153]]]
        - [3, [[197,3],[238,2],[238,78],[201,103]]]
        - [4, [[212,107],[224,240],[369,110],[377,9]]]
        - [5, [[262,240],[254,283],[222,310],[233,395],[200,434],[252,470],[277,433],[277,357],[311,320],[302,393],[344,431],[412,320],[370,297],[391,190],[367,171],[289,253]]]
        - [6, [[367,171],[424,112],[444,124],[420,219],[459,239],[412,320],[370,297],[391,190]]]
        - [7, [[424,112],[461,73],[477,82],[458,167],[488,185],[459,239],[420,219],[444,124]]]
        - [8, [[461,73],[490,42],[507,53],[486,128],[512,148],[488,185],[458,167],[477,82]]]
    2:
        input_video: "rtsp://admin:Perfumeriasunidas2!@192.168.0.102:554/cam/realmonitor?channel=2&subtype=0"
        # input_video: "camera_2_video.mp4"
        output_video: "camera_2_processed.mp4"
        camera_sn: "AD0B109PAZ473AF-2"
        polygons:
        - [9, [[2,111],[38,61],[100,338],[149,324],[158,473],[76,476],[3,338]]]
        - [10, [[38,61],[101,36],[149,324],[100,338]]]
        - [11, [[101,36],[163,18],[200,310],[149,324]]]
        - [12, [[163,18],[227,3],[252,296],[200,310]]]
        - [13, [[422,8],[406,210],[382,214],[441,312],[484,299],[538,32],[471,2]]  ]  
        - [14, [[441,312],[547,470],[598,470],[700,260],[701,128],[538,32],[484,299]]]
    3:
        input_video: "rtsp://admin:Perfumeriasunidas2!@192.168.0.102:554/cam/realmonitor?channel=3&subtype=0"
        # input_video: "camera_3_video.mp4"
        output_video: "camera_3_processed.mp4"
        camera_sn: "AD0B109PAZ473AF-3"
        polygons:
        - [15, [[97,308],[140,435],[145,468],[253,395],[234,386],[210,252]]]
        - [16, [[253,395],[324,348],[308,335],[308,203],[210,252],[234,386]]]
        - [17, [[324,348],[393,300],[375,290],[388,162],[308,203],[308,335]]]
        - [18, [[393,300],[444,266],[431,259],[448,141],[388,162],[375,290]]]
        - [19, [[297,406],[511,243],[530,254],[495,370],[378,476],[293,476]]]
        - [20, [[556,40],[620,74],[661,58],[702,87],[702,119],[598,308],[580,290],[549,317],[528,304],[530,254],[511,243]]]
        - [21, [[311,14],[384,2],[374,160],[302,193]]]
        - [22, [[374,160],[430,140],[445,3],[384,2]]]
    4:
        input_video: "rtsp://admin:Perfumeriasunidas2!@192.168.0.102:554/cam/realmonitor?channel=4&subtype=0"
        # input_video: "camera_4_video.mp4"
        output_video: "camera_4_processed.mp4"
        camera_sn: "AD0B109PAZ473AF-4"
        polygons:
        - [23, [[260,433],[453,376],[495,471],[628,470],[523,287],[250,345]]]
        - [24, [[225,7],[233,109],[353,87],[342,7]]]

Verify the username, password, IP and camera number of the RSTP URL. Also, add the corresponding polygons and their ID if changes are made to the zones in the future.

## Test the algorithm in the terminal

Run the Python script to verify the correct configuration. Note that the script receives one argument: the camera number. In this case we'll try the 2nd camera.

    python3 process_test.py 2

## Add autostart

Close the last Python script. Create the service file. Keep in mind that a service will be created for each camera, in this case we will create one for camera 4

    sudo nano /etc/systemd/system/python-script-cam4.service

Copy this lines:

    [Unit]
    Description=Python Script in Docker Container
    Requires=docker.service
    After=docker.service

    [Service]
    User=jorinbriq01
    Type=simple
    ExecStartPre=/bin/bash -c 'while ! docker ps | grep briq_container; do sleep 1; done'
    ExecStart=/usr/bin/docker exec briq_container /bin/bash -c "cd /unidas_system/files/torch2trt/ && python3 process_test.py 4"
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target

Check the container, user and camera number in the Execstart. Then, run this in terminal:

    sudo systemctl daemon-reload
    sudo systemctl enable python-script-cam4.service
    sudo systemctl start python-script-cam4.service

If there are errors during service startup, check the logs and correct any script or system configuration errors.

    sudo journalctl -u python-script-cam4.service -f

If there are no errors, check the service status

    sudo systemctl status python-script-cam4.service
