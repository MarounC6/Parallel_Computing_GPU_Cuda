#!/bin/sh
PATH=/usr/bin:/bin:/usr/sbin:/sbin;

usage() {
    echo "Usage: $0 <action> <eclipse_dir>"
    echo "    <action> : 'install' or 'uninstall'"
    echo "    <eclipse_dir> : eclipse installation directory"
    exit 1;
}

if [ "$#" -ne 2 ]; then
    echo "Illegal number of arguments"
    usage;
fi

action=$1;
eclipse_dir=$2;

if [ ! -d $eclipse_dir ]; then
    echo "Eclipse directory '$eclipse_dir' not found";
    usage;
fi

eclipse_binary=$eclipse_dir/eclipse;

if [ ! -f "$eclipse_binary" -o ! -x "$eclipse_binary" ]; then
    echo "Eclipse binary not found in eclipse directory at:";
    echo "$eclipse_binary";
    usage;
fi

IUs="com.nvidia.cuda.feature.group,com.nvidia.cuda.remote.feature.feature.group,com.nvidia.cuda.docker.feature.feature.group";

case "$action" in
    uninstall)
        echo "Uninstalling Nsight EE plugins from $eclipse_dir...";
        $eclipse_binary -clean -purgeHistory -application org.eclipse.equinox.p2.director -noSplash -uninstallIUs $IUs
        echo "Finished uninstallation of Nsight EE plugins."
        ;;
    install)
        PLUGINS_ZIP=`dirname $0`/../nsightee_plugins/com.nvidia.cuda.repo-1.0.0-SNAPSHOT.zip;
        echo "Installing Nsight EE plugins to $eclipse_dir...";
        $eclipse_binary -clean -purgeHistory -application org.eclipse.equinox.p2.director -noSplash -repository http://download.eclipse.org/releases/oxygen,jar:file:$PLUGINS_ZIP!/ -installIUs $IUs
        echo "Finished installation of Nsight EE plugins."
        ;;
    *)
        echo "Action '$action' is invalid"
        usage;
        ;;
esac
