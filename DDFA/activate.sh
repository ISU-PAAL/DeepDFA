#!/bin/bash
which module >> /dev/null && module load openjdk git gcc/10.2.0-zuvaafu cuda/11.3.1-z4twu5r
rootdir="$PWD"
envdir="$rootdir/venv"
source $envdir/bin/activate

export PYTHONPATH=$rootdir
export PATH="$PATH:$rootdir/storage/external/joern/joern-cli"
export TS_SOCKET="/tmp/socket.linevd"
