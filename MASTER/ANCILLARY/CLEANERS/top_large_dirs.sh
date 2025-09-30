sudo find /home -xdev -type d -links 2 -print0 \
| sudo xargs -0 du -sk 2>/dev/null \
| sort -nrk1 | head -30 \
| awk '{ printf "%8.2f GiB\t%s\n", $1/1024/1024, substr($0, index($0,$2)) }'
