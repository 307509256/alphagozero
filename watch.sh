#!/bin/bash
name="./alphagozero.sh"
dir="/root/gongjia/alphagozero"
while true
 do
        c=`ps x|grep $name|grep -v grep|wc -l`
        if [ $c -eq 0 ]
        then
         cd $dir
         $name
        fi
        sleep 10
done
