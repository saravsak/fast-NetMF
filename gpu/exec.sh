#!/bin/bash
./clean.sh
./make.sh
./netmf_${1}
./clean.sh
