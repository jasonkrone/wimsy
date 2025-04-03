
function install_conda() {
    wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
    export PATH="$HOME/anaconda3/bin:$PATH"

    source bash/setup.sh ; activate_or_create_conda_env train ./requirements/requirements.txt
}

function get_data() {

    mkdir -p /fsx/books
    sudo chmod go+rw /fsx/books
    sudo chmod go+rw /fsx
    source keys.env ; aws s3 cp --recursive s3://jpt-data/dolma/tokenized/books/ /fsx/books/
}

function setup_git() {

    conda install gh --channel conda-forge
}

function setup_enroot() {
    aws s3 cp s3://jpt-apps/jpt.sqsh .

}


function run_docker() {

    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -it jpolingkrone/jpt:latest -v /home/ubuntu/sky_workdir:/sky_workdir

}

function get_books_data() {

    mkdir /home/ubuntu/data
    source /home/ubuntu/sky_workdir/jpt/keys.env ; aws s3 sync s3://books-memmap-decontaminated-ctx-4096-tokenizer-tiktoken /home/ubuntu/data/books-memmap-decontaminated-ctx-4096-tokenizer-tiktoken


}


function rsync_code() {
    for ip_addr in "$@"; do
        echo "syncing to ${ip_addr}"
        rsync -avz -e "ssh -i ~/.ssh/sky-key" ~/Developer/jpt ubuntu@$ip_addr:~/sky_workdir && rsync -avz -e "ssh -i ~/.ssh/sky-key" ~/Developer/jpt ubuntu@129.213.91.110:~/sky_workdir
    done
}

function fix_docker() {

    sudo usermod -aG docker $USER
    newgrp docker

}