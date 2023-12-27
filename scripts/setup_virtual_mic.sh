#!/bin/sh

# After running this setup script you can redirect audio to the
# virtual microphone for example by running something like
#
#     PULSE_SINK=VirtMicSource mpv foo.wav


pactl load-module module-null-sink sink_name=VirtMicSource sink_properties=device.description="VirtMicSource"
pactl load-module module-virtual-source source_name=VirtualMic master=VirtMicSource.monitor

# The following commands are only necessary when we want to redirect any input
# to both the 'normal' sound output and the virtual microphone.
# If we don't do this we have to manually set the output of the application
# we want to redirect to the virtual microphone to 'Source'
#pactl load-module module-loopback source=Source.monitor sink=[name field of your Built-in Audio Analog stereo]
#pactl set-default-sink Source
