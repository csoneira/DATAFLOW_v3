function [status, result] = sendbashEmail2(subject,to,message,attachment)
%!/bin/bash
%sudo apt-get install sendemail
%sudo apt-get install libnet-ssleay-perl
%sudo apt-get install libnet-smtp-ssl-perl
%if error change:
%/usr/bin/sendemail on line 1907: 'SSLv3 TLSv1' => 'SSLv3' 
%if persistent do:
%if (! IO::Socket::SSL->start_SSL($SERVER, SSL_version => 'SSLv23:!SSLv2', SSL_verify_mode => 0)) {

b                   =                                                                                              ' ';
from                =                                                           '-S from="rpc.slow.control@gmail.com"';
smtpOptions         = '-S smtp-use-starttls -S ssl-verify=ignore -S smtp-auth=login -S smtp=smtp://smtp.gmail.com:587';
user                =                                                   '-S smtp-auth-user=rpc.slow.control@gmail.com';
passWD              =                                                         '-S smtp-auth-password=bgylttrezpuauuho';

subject             =                                                                             ['-s "' subject '"'];
message             =                                                                             ['"' message '"'];

sendTo = [to{1}];
for i=2:size(to,2)
    sendTo = [sendTo ','  to{i}];
end

sendAttachment = [''];
for i=1:size(attachment,2)
    sendAttachment = [sendAttachment b '-a' b attachment{i}];
end

%This is the old comand using sendEmail
%[status, result] = system(['sendEmail -o tls=yes ' from ' ' sendTo ' -s smtp.gmail.com:587 -xu rpc.slow.control@gmail.com -xp bgylttrezpuauuho -u "' subject '" -m "' message '" ' sendAttachment]);


[status, result] = system(['echo' b message  b '| mail' b sendAttachment b subject b smtpOptions b user b passWD b sendTo]);
%Thsi is an example
%echo "my test"  | mail  -a java.log.10387 -s "this is a prove" -S smtp-use-starttls -S ssl-verify=ignore -S smtp-auth=login -S smtp=smtp://smtp.gmail.com:587 -S from="rpc.slow.control@gmail.com" -S smtp-auth-user=rpc.slow.control@gmail.com -S smtp-auth-password=bgylttrezpuauuho -S verbose alberto@coimbra.lip.pt
