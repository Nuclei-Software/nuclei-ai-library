BUILDTIME=$(date +%Y%m%d_%H%M)

LOGDIR=exported
mkdir -p $LOGDIR
PERFLOG=perf.log

RUNNER=${RUNNER:-cycm}
OLV=${OLV:-Ofast}
VLEN=${VLEN:-128}
CORE=${CORE:-n900fd}
DOWNLOAD=${DOWNLOAD:-ilm}
TMOUT=${TMOUT:-60}
AUTOVEC=${AUTOVEC-1}

if [ -f Makefile ] ; then
    APPNAME=$(cat Makefile | grep TARGET | cut -d "=" -f2|tr -d ' ')
fi
if [ "x$APPNAME" == "x" ] ; then
    echo "Unable to perf this app!"
    exit 1
fi

MAINFILES="main.c Makefile toolchain_*.mk perf.sh perf.csv perf.py patch.diff $PERFLOG"
ZIPFILES="$MAINFILES ${APPNAME}.dasm ${APPNAME}.elf ${APPNAME}.dump"
PERFZIP=$LOGDIR/${APPNAME}_${CORE}_${DOWNLOAD}_${OLV}_${BUILDTIME}.zip

if [[ "$CORE" == *"x"* ]] ; then
    QEMUCMD="qemu-system-riscv64 -M nuclei_evalsoc,download=$DOWNLOAD -cpu nuclei-nx900fd,ext=v_zba_zbb_zbc_zbs_zfh_zvfh_zvl${VLEN}b_xxldspn3x "
    if [[ "$VLEN" -gt 256 ]] ; then
        CYCM=${CYCM:-ni900_best_config_vlen${VLEN}_thread12_cymodel}
    else
        CYCM=${CYCM:-ux900_best_config_vlen${VLEN}_thread12_cymodel}
    fi
else
    QEMUCMD="qemu-system-riscv32 -M nuclei_evalsoc,download=$DOWNLOAD -cpu nuclei-n900fd,ext=_zba_zbb_zbc_zbs_zfh_zvfh_zve32f_xxldspn3x "
    CYCM=${CYCM:-n900_best_config_vlen${VLEN}_thread12_cymodel}
fi

COMMON_OPTS="AUTOVEC=$AUTOVEC OLV=$OLV CORE=$CORE DOWNLOAD=$DOWNLOAD"

# SIMU is required to exit when error happened
if [ "x$RUNNER" == "xqemu" ] ; then
    COMMON_OPTS="${COMMON_OPTS} SIMU=qemu"
else
    COMMON_OPTS="${COMMON_OPTS} SIMU=xlspike"
fi

function setup_terapines() {
    local ver=${1:-latest}
    export OLDPATH=$PATH
    export PATH=/home/share/devtools/zcc/linux64/$ver/bin/:$PATH
    echo "Using Terapines toolchain $(realpath `which zcc`)"
}

function setup_nuclei() {
    local ver=${1:-latest}
    export OLDPATH=$PATH
    export PATH=/home/share/devtools/toolchain/nuclei_gnu/linux64/newlibc/$ver/gcc/bin/:$PATH
    echo "Using Nuclei toolchain $(realpath `which riscv64-unknown-elf-gcc`)"
}

function reset_env() {
    export PATH=$OLDPATH
}

function run_app() {
    local elf=${1-*.elf}
    if [ "x$RUNNER" == "xxlspike" ] ; then
        xl_spike $elf | tee -a $PERFLOG
    elif [ "x$RUNNER" == "xqemu" ] ; then
        timeout --foreground ${TMOUT}s $QEMUCMD -smp 1 -icount shift=0 -nodefaults -nographic -serial stdio -kernel $elf | tee -a $PERFLOG
    else
        $CYCM --cycle 0 $elf | tee -a $PERFLOG
    fi
}

function perf_app() {
    local toolchain=$1
    local ext=$2
    local toolver=${3-latest}

    export SILENT=1

    makeopts="${COMMON_OPTS} TOOLCHAIN=$toolchain ARCH_EXT=$ext "
    make $makeopts clean
    set -x
    if make $makeopts -j dasm; then
        command cp -f ${APPNAME}.elf ${APPNAME}_${toolchain}${ext}.elf
        echo "PERFCSV,$toolver,$toolchain,$ext,PASS" >> $PERFLOG
        echo "Build command: make $makeopts clean dasm" >> $PERFLOG
        make $makeopts showtoolver >> $PERFLOG 2>&1
        make $makeopts showflags >> $PERFLOG 2>&1
        backelf=${APPNAME}_${toolver}_${toolchain}${ext}.elf
        command cp -f ${APPNAME}.elf ${backelf}
        run_app ${APPNAME}.elf
        zip -u ${PERFZIP} ${backelf} $MAINFILES
        mv -f ${backelf} $LOGDIR/
    else
        echo "PERFCSV,$toolver,$toolchain,$ext,FAIL" >> $PERFLOG
    fi
    set +x
}

rm -f $PERFLOG
touch $PERFLOG

export OLV=$OLV
git diff > patch.diff
echo "CFGCSV,CORE=${CORE},DOWNLOAD=${DOWNLOAD},OLV=${OLV},RUNNER=${RUNNER},SDKVER=$(git log -1 --oneline),PATCH=patch.diff" >> $PERFLOG
#set -x

for toolver in gcc13 gcc14; do
    setup_nuclei $toolver
    for toolchain in nuclei_gnu nuclei_llvm; do
        for ext in _zfh_zvfh_zve32f_zvl${VLEN}b; do
            perf_app $toolchain $ext $toolver
        done
    done
    reset_env
done

if [ -f ~/zcc_lic.sh ] ; then
    ~/zcc_lic.sh
fi

for toolver in lite pro; do
    setup_terapines $toolver
    for toolchain in terapines; do
        for ext in _zfh_zvfh_zve32f_zvl${VLEN}b; do
            perf_app $toolchain $ext $toolver
        done
    done
    reset_env
done

echo "See ${PERFLOG} for detailed result"
cat ${PERFLOG} | grep -a CSV

if [ -f perf.py ] ; then
    python perf.py
    echo "Please find the processed data in perf.csv"
fi

echo "Check the final zip in $PERFZIP"
