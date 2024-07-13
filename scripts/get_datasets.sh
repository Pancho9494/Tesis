#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

get_3DMatch() {

    declare -A fragments=(
        ["redkitchen"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen.zip"
        ["home_at_scan1_2013_jan_1"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1.zip"
        ["home_md_scan9_2012_sep_30"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30.zip"
        ["scan3"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3.zip"
        ["maryland_hotel1"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1.zip"
        ["maryland_hotel3"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip"
        ["mit_76_studyroom/76-1studyroom2"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2.zip"
        ["lab_hj_tea_nov_2_2012_scan1_erika"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip"
    )

    declare -A evaluation=(
        ["redkitchen"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen-evaluation.zip"
        ["home_at_scan1_2013_jan_1"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1-evaluation.zip"
        ["home_md_scan9_2012_sep_30"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30-evaluation.zip"
        ["scan3"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3-evaluation.zip"
        ["maryland_hotel1"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1-evaluation.zip"
        ["maryland_hotel3"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3-evaluation.zip"
        ["mit_76_studyroom/76-1studyroom2"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2-evaluation.zip"
        ["lab_hj_tea_nov_2_2012_scan1_erika"]="http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation.zip"
    )

    mkdir -p ../data/3DLoMatch
    for i in "${!fragments[@]}"
    do
        # echo "Downloading $i ${fragments[$i]} ${evaluation[$i]}"

        # TODO: maybe extract this into a function? maybe its not even worth it
        mkdir -p ../data/3DLoMatch/fragments/$i
        wget "${fragments[$i]}" --directory-prefix=../data/3DLoMatch/fragments/$i --continue -O ../data/3DLoMatch/fragments/$i/$i.zip
        unzip -qq ../data/3DLoMatch/fragments/$i/$i.zip -d ../data/3DLoMatch/fragments/$i
        rm -f ../data/3DLoMatch/fragments/$i/$i.zip
        temp_folder="$(ls $SCRIPT_DIR/../data/3DLoMatch/fragments/$i/)"
        echo $temp_folder
        mv ../data/3DLoMatch/fragments/$i/$temp_folder/* ../data/3DLoMatch/fragments/$i/
        rm -rf ../data/3DLoMatch/fragments/$i/$temp_folder
        

        mkdir -p ../data/3DLoMatch/evaluation/$i
        wget "${evaluation[$i]}" --directory-prefix=../data/3DLoMatch/evaluation/$i --continue -O ../data/3DLoMatch/evaluation/$i/$i.zip
        unzip -qq ../data/3DLoMatch/evaluation/$i/$i.zip -d ../data/3DLoMatch/evaluation/$i
        rm -f ../data/3DLoMatch/evaluation/$i/$i.zip
        temp_folder="$(ls $SCRIPT_DIR/../data/3DLoMatch/evaluation/$i/)"
        echo $temp_folder
        mv ../data/3DLoMatch/evaluation/$i/$temp_folder/* ../data/3DLoMatch/evaluation/$i/
        rm -rf ../data/3DLoMatch/evaluation/$i/$temp_folder
        
        echo "" 
    done

    echo "Please visit https://github.com/WHU-USI3DV/WHU-TLS for the WHU-TLS dataset"
}

get_3DMatch