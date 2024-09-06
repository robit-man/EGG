#!/bin/bash
# dummy_script.sh
# Change 'password' to your system password
echo "Caching sudo credentials..."
echo password | sudo -S -v
echo "Credentials cached."
