#!/bin/env bash

# See also: https://erda.dk/wsgi-bin/setup.py?topic=sftp

if [ -f "${DLC_ERDA_KEY}" ]
then
	mkdir -p ${DLC_ERDA_MOUNT}
	sshfs ${DLC_ERDA_USER}@io.erda.dk:${DLC_ERDA_DIR} ${DLC_ERDA_MOUNT} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${ERDA_KEY}
else
	echo "'${DLC_ERDA_KEY}' is not an SSH key"
fi
