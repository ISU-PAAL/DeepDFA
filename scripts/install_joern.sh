# https://docs.joern.io/installation/
# Version 1.1.107 was used for the paper
mkdir joern
cd joern
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o joern-install.sh
chmod u+x joern-install.sh
# now run the script with the following options
printf "y\\n$PWD/joern\\nn\\nv1.1.107\\n" | ./joern-install.sh --interactive --without-plugins
