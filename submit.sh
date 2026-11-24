#!/bin/bash
# Copy the code to the LIACS cluster and submit cluster.sbatch (any L40s node). Run from your laptop:
#     bash submit.sh
# Needs the `l40s` alias in ~/.ssh/config (fs.cc.liacs.nl via ssh.liacs.nl).
set -euo pipefail
cd "$(dirname "$0")"

REMOTE=l40s
DIR=fmai-ass1                                    # folder in your cluster home; job logs end up in ~/$DIR/logs

# WSL's ssh doesn't read the Windows ~/.ssh (config + keys); fall back to Windows' ssh.exe, which does.
SSH=ssh
if ! ssh -G "$REMOTE" 2>/dev/null | grep -qi "^hostname fs.cc.liacs.nl"; then
    SSH=/mnt/c/Windows/System32/OpenSSH/ssh.exe  # not Git's ssh.exe: it can't run the config's Windows ProxyCommand
fi
echo "Using $SSH"

tar czf - --exclude=.venv --exclude=results --exclude=logs --exclude=__pycache__ . \
    | "$SSH" "$REMOTE" "mkdir -p ~/$DIR && tar xzf - -C ~/$DIR && cd ~/$DIR && sbatch ${SBATCH_ARGS:-} cluster.sbatch"
echo "Follow with: $SSH $REMOTE 'squeue -u \$USER; tail -n 20 ~/$DIR/fmai-*.out'"
