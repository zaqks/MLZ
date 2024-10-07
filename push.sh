#!/bin/bash

find . | grep "__pycache__$" | xargs rm -r

git add .
git commit -am $0
git push


