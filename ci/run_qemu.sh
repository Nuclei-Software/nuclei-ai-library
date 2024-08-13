#!/usr/bin/env bash

# please run in <app> folder
# assume directory structure as below:
# - nuclei-ai-library
# - nuclei-sdk
LOGDIR=${1:-logs}

if [ "x${NUCLEI_SDK_ROOT}" == "x" ] ; then
    NUCLEI_SDK_ROOT=$(readlink -f ../nuclei-sdk)
fi

NSDK_BENCH_CLI=${NUCLEI_SDK_ROOT}/tools/scripts/nsdk_cli/nsdk_bench.py

if [ ! -f ${NSDK_BENCH_CLI} ] ; then
    echo "ERROR: Can't find ${NSDK_BENCH_CLI}, please check whether your NUCLEI_SDK_ROOT environment variable is set correctly!"
    exit 1
fi

if [ ! -f ci/app.json ] ; then
    echo "ERROR: Please run script in case folder, not in the ci folder or other places!"
    echo "INFO: eg. ./ci/run_qemu.sh"
    exit 1
fi

if [ -d $LOGDIR ] ; then
    echo "WARN: Log directory $LOGDIR already exists, will remove it in 3 seconds!"
    sleep 3
    rm -rf $LOGDIR
    echo "INFO: $LOGDIR removed!"
fi

if [ -f /home/share/devtools/env.sh ] ; then
    echo "INFO: setup run environment for you!"
    source /home/share/devtools/env.sh
    activate_swdev
fi

echo "INFO: Start to run case examples on qemu for rv32 and rv64 cores!"
runcmd="python3 ${NSDK_BENCH_CLI} --appcfg ci/app.json --hwcfg ci/qemu.json --logdir ${LOGDIR} --run_target qemu --run --parallel=\"-j\""
echo "INFO: run command: $runcmd"
$runcmd
ret=$?

retmsg="PASS"
if [ "x$ret" != "x0" ] ; then
    retmsg="FAIL"
fi
echo "INFO: Run on qemu $retmsg, check logs via httpserver_cli -d $LOGDIR"
exit $ret
