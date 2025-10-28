function dev = locatedev(locatedev,conf)


for i = 1:length(conf.dev)
    if strcmp(locatedev,[conf.dev(i).name conf.dev(i).subName])
        dev = i;
        return
    end
end
return