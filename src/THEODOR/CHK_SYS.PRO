function chk_sys

; checks wether or not the operating system is PC or UNIX based

if strpos(!dir,'/') eq -1 then sys='PC' else sys='UNIX'

return, sys

end
