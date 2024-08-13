#!/bin/bash

LOGDIR=${LOGDIR:-logs/ci}
CONFIG=${CONFIG:-ux900}
DRYRUN=${DRYRUN:-0}
TMOUT=${TMOUT:-0}

RUNYAML=req_runners.yaml

# check if exist RUNYAML, release it first
if [ -f ${RUNYAML} ] ; then
    runboard --release ${RUNYAML}
    rm -f ${RUNYAML}
fi

if [ "x$CONFIG" == "x" ] ; then
    echo "No config $CONFIG specified, exit"
    exit 1
fi

reqboards=""
if [[ "$CONFIG" == *"200"* ]] ; then
    reqboards="ddr200t,${reqboards}"
fi
if [[ "$CONFIG" == *"600"* ]] || [[ "$CONFIG" == *"300"* ]] ; then
    reqboards="ku060,${reqboards}"
fi
if [[ "$CONFIG" == *"900"* ]] ; then
    reqboards="vcu118,${reqboards}"
fi

if [ "x${NUCLEI_SDK_ROOT}" == "x" ] ; then
    NUCLEI_SDK_ROOT=$(readlink -f ../nuclei-sdk)
fi

NSDK_RUNNER_CLI=${NUCLEI_SDK_ROOT}/tools/scripts/nsdk_cli/nsdk_runner.py

if [ ! -f ${NSDK_RUNNER_CLI} ] ; then
    echo "ERROR: Can't find ${NSDK_RUNNER_CLI}, please check whether your NUCLEI_SDK_ROOT environment variable is set correctly!"
    exit 1
fi

runboard --request "${reqboards}#${TMOUT}"

if [ -f ${RUNYAML} ] ; then
    echo "Requested required boards ${reqboards}"
else
    echo "No boards requested!"
    exit 1
fi

runcmd="python3 ${NSDK_RUNNER_CLI} --appyaml ci/app.yaml --runyaml ${RUNYAML} --logdir ${LOGDIR} --runon fpga --config ${CONFIG}"

if [ "x${CI_PIPELINE_ID}" != "x" ] ; then
    runcmd="$runcmd --uniqueid pipeline${CI_PIPELINE_ID}"
fi

echo $runcmd
ret=0
if [ "x$DRYRUN" == "x0" ] ; then
    if [ "x${NUCLEI_SDK_ROOT}" == "x" ] ; then
        echo "NUCLEI_SDK_ROOT is not set in environment variables"
        exit 1
    fi
    if [ -d ${LOGDIR} ] ; then
        echo "Remove previously existed log directory ${LOGDIR}"
        rm -rf ${LOGDIR}
    fi
    eval $runcmd
    ret=$?
fi

retmsg="PASS"
if [ "x$ret" != "x0" ] ; then
    retmsg="FAIL"
fi

runboard --release ${RUNYAML}

echo "INFO: Run on fpga for $CONFIG : $retmsg, check log directory in $(readlink -f ${LOGDIR})"
exit $ret
