#!/bin/bash

find . | grep -e "__pycache__$" | xargs rm -r