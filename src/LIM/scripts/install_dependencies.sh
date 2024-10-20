#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(builtin cd $SCRIPT_DIR/..; pwd)

activate_venv() {
    echo "Activating local python version"
    if [ ! -e "$BASE_DIR/.localPython" ]; then
        wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz -O python3.8.5.tgz
        tar xzvf python3.8.5.tgz
        mkdir $BASE_DIR/.localPython
        cd $SCRIPT_DIR/Python-3.8.5
        ./configure  --prefix=$BASE_DIR/.localPython --enable-optimizations
        make
        sudo make install
        cd $SCRIPT_DIR
        sudo rm -rf Python-3.8.5
        rm python3.8.5.tgz
    fi
}

install_dependencies() {
    $BASE_DIR/.localPython/bin/pip3 install -r $BASE_DIR/requirements.txt
    
    # TODO this should loop over each dir inside lib/ and search for requitements files
    $BASE_DIR/.localPython/bin/pip3 install -r $BASE_DIR/lib/OverlapPredator/requirements.txt

    if [ ! -e "$BASE_DIR/data/weights/predator" ]; then
        sh $BASE_DIR/lib/OverlapPredator/scripts/download_data_weight.sh
        mkdir -p $BASE_DIR/data/weights/predator
        mv $SCRIPT_DIR/weights/* $BASE_DIR/data/weights/predator/
        rm -rf $SCRIPT_DIR/data $SCRIPT_DIR/weights
    fi
}

create_submodules_symlinks() {
    for module in $BASE_DIR/../submodules/*/
    do
        echo $module
        touch $module/"__init__.py"
        # find $module -depth -maxdepth 2 -name "__init__.py" 
    done
}

# activate_venv
# install_dependencies
create_submodules_symlinks
