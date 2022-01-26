#!/bin/bash
set -eux

DVC_REMOTE_DEFAULT=$(dvc remote default)

[ ! -e .dvc/config.local ] && {
    dvc remote modify --local $DVC_REMOTE_DEFAULT user $DVC_REMOTE_USER
    dvc remote modify --local $DVC_REMOTE_DEFAULT password $DVC_REMOTE_PASS
}

# dvc pull --run-cache
# dvc repro $@
dvc exp run $@
