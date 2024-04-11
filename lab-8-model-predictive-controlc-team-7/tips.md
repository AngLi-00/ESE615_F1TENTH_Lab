### Use ssh to connnect with car:

`ssh nvidia@<car ip address>`


`ssh nvidia@172.20.10.12` (凉)

`ssh nvidia@10.103.150.121` (AirPennNet)

### Remote Development using SSH

[Install the Remote-SSH extension](https://code.visualstudio.com/docs/remote/ssh)

### Use ROS DOMAIN to communicate with(rviz on local machine):

*Use Hotspot instead of AirPennNet when you want to use ROS DOMAIN to communicate*

`export ROS_DOMAIN_ID=<your ros domain id>`

in ~/.bashrc on both local machine and car:

`export ROS_DOMAIN_ID=77`

### tmux:

`ctrl + b` followed by `%`     : Split screen left and right

`ctrl + b` followed by `“`     : Split screen up and down

`ctrl + b` followed by `[`     : readmode (`q`: quit)

`ctrl + b` followed by `o`     : next window

`ctrl + b` followed by `q`  then choose a number      : switch to one specific window

(out of tmux window) tmux kill-server : kill all tmux windows

*Switch pane by mouse:*

First open tmux in terminal

    tmux

Enter tmux's command mode by pressing `Ctrl`+`b` followed by `:`

Then type the following command and press Enter to enable mouse support:

    set -g mouse on