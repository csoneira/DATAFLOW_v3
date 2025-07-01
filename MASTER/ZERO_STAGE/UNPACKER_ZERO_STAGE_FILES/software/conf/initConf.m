%% Check system
[status, result] = system('hostname');result = result(1:end-1);
if(strcmp(result,'manta'))
    OS = 'linux';
    HOSTNAME    = 2;
    SYSTEMNAME  = 'mingo01';
    %Software location
    HOME        = ['/home/alberto/gate/localDocs/lip/daqSystems/' SYSTEMNAME '/'];
    %System data structure
    SYS         = [HOME 'system/'];
    INTERPRETER = 'matlab';
else
    more off
    warning('off');%,' Matlab-style short-circuit operation performed for operator &');
    OS = 'linux';
    HOSTNAME    = 1;
    SYSTEMNAME  = 'mingo01';
    %Software location
    HOME        = '/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/';
    %System data structure
    SYS         = [HOME 'system/'];
    INTERPRETER = 'octave';
end
