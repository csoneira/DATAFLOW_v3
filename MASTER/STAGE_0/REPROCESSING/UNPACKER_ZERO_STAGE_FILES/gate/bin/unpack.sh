cd $HOME/software/

# Save current HOME
#OLD_HOME="$HOME"

# Set temporary HOME
#export HOME="/media/externalDisk/gate"

octave --no-gui --no-history $HOME/software/unpackingContinuous.m

# Restore original HOME
#export HOME="$OLD_HOME"
