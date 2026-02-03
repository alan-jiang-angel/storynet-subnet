#!/bin/sh

# miner-run.sh -w <WALLET_NAME> -h <HOTKEY_NAME> -a <AXON_PORT> -l <LOG_LEVEL>

export OPENAI_API_KEY=sk-proj-...

WALLET_NAME="miner"
HOTKEY_NAME="default"
NETUID=92
AXON_PORT=8091
LOG_LEVEL="info"

while getopts w:h:a:n:l: flag
do
    case "${flag}" in
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        a) AXON_PORT=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
        n) MINER_NAME=${OPTARG};;
    esac
done

EXTERNAL_IP="65.108.13.250"
AXON_EXTERNAL_PORT=${AXON_PORT}

# echo "python neurons/miner.py --netuid ${NETUID} --wallet.name ${WALLET_NAME} --wallet.hotkey ${HOTKEY_NAME} --subtensor.network finney --axon.port ${AXON_PORT} --axon.external_ip ${EXTERNAL_IP} --axon.external_port ${AXON_EXTERNAL_PORT} --logging.${LOG_LEVEL}"
pm2 start "python neurons/miner.py --netuid ${NETUID} --wallet.name ${WALLET_NAME} --wallet.hotkey ${HOTKEY_NAME} --subtensor.network finney --axon.port ${AXON_PORT} --axon.external_ip ${EXTERNAL_IP} --axon.external_port ${AXON_EXTERNAL_PORT} --logging.${LOG_LEVEL}" --name sn${NETUID}-${HOTKEY_NAME}
pm2 log sn${NETUID}-${HOTKEY_NAME}
