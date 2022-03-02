#!/bin/bash
set -eux

# uncomment for use with webdav remote
#DVC_REMOTE_DEFAULT=$(dvc remote default)

#[ ! -e .dvc/config.local ] && {
#    dvc remote modify --local $DVC_REMOTE_DEFAULT user $DVC_REMOTE_USER
#    dvc remote modify --local $DVC_REMOTE_DEFAULT password $DVC_REMOTE_PASS
#}

dvc exp run $@
