#!/bin/bash

function run_ngrams() { 

    ## get elastic search running
    compose_path=./docker/data_prep/docker_compose.yaml
    docker compose -f $compose_path up -d
    sleep 15

    ## run data prep
    config_path=./configs/preprocess/preprocess_starcoder.yaml
    docker compose -f $compose_path exec jpt_prep_data bash -c "cd /sky_workdir && python -m preprocessing.prep_data --config $config_path"
    docker compose -f $compose_path down

}


function delete_index() {
    compose_path=./docker/data_prep/docker_compose.yaml
    docker compose -f $compose_path exec jpt_prep_data bash -c "python -c 'from elasticsearch import Elasticsearch; Elasticsearch([\"http://elasticsearch:9200\"]).indices.delete(index=\"$1\")'"
}

function run_starcoder() {
    source keys.env; docker login -u jpolingkrone -p $DOCKER_PASSWORD

    ## get elastic search running
    compose_path=./docker/data_prep/docker_compose.yaml
    docker compose -f $compose_path up -d
    sleep 15

    ## run data prep
    config_path=./configs/preprocess/preprocess_starcoder.yaml
    docker compose -f $compose_path exec jpt_prep_data bash -c "cd /sky_workdir && python -m preprocessing.prep_data --config $config_path"
    docker compose -f $compose_path down

}

function run_docker() {
    # /tmp/.cache
    # sudo rm /etc/enroot/hooks.d/98-nvidia.sh
    #sudo mkdir -p /local_out
    #docker run -it -v /es:/es -v /local_out:/local_out -v ~/sky_workdir:/sky_workdir -v /output:/output jpt:elastic bash
    #docker-compose -f ./docker/data_prep/docker_compose.yaml up
    # jupyter lab --ip=0.0.0.0 --port=9999 --no-browser --allow-root

    source keys.env; docker login -u jpolingkrone -p $DOCKER_PASSWORD
    compose_path=./docker/data_prep/docker_compose.yaml
    docker compose -f $compose_path up -d
    #sleep 15
    #docker compose -f $compose_path exec jpt_prep_data bash
    docker compose -f $compose_path exec jpt_prep_data bash -c "cd /sky_workdir && pytest ./tests/test_score_filter.py"
    #docker compose -f $compose_path exec jpt_prep_data bash -c "cd /sky_workdir && python -m preprocessing.prep_data --config ./configs/preprocess/dclm_top500b.yaml"
    # TODO: there's a timing thing to think about
}

function make_dolma() {
    cd ~/sky_workdir/dolma
    maturin develop
    pip install tokenizers==0.20.0
    cd ~/sky_workdir
}


function run_replacer_test() {
    source keys.env; python -m preprocessing.prep_data --config ./temp/temp_span_config.yaml
    #zcat /home/ubuntu/sky_workdir/temp/test_docs_to_decontam.jsonl.tmp > /home/ubuntu/sky_workdir/temp/todiff.jsonl
    diff /home/ubuntu/sky_workdir/temp/test_docs_to_decontam.jsonl.tmp /home/ubuntu/sky_workdir/temp/test_docs_out.jsonl
}


function run_recover() {
    # TODO: this is set for pes2o inputs
    config_path=./configs/decontaminate/wiki_decontaminate.yaml
    source keys.env; python -m preprocessing.recover_ids_and_line_nums --config $config_path
}

function run_dclm() {
    config_path=./configs/decontaminate/dclm_decontaminate.yaml
    source keys.env; python -m preprocessing.prep_data --config $config_path
}


function run_jupyter() {
    jupyter notebook --no-browser --port=9999
}


function make_nl_filter() {
    source keys.env; python -m preprocessing.prep_data --config ./configs/decontaminate/make_natural_language_filter.yaml
}


function make_code_filter() {
    source keys.env; python -m preprocessing.prep_data --config ./configs/decontaminate/make_code_filter.yaml
}


function run_code_filter() {
    source keys.env; python -m preprocessing.prep_data --config ./configs/decontaminate/apply_code_filter.yaml
}


function run_decontam() {
    source keys.env; python -m preprocessing.prep_data --config ./configs/decontaminate/wiki_decontaminate.yaml
}


function run_es() {
    docker run -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node"  -v /es:/usr/share/elasticsearch/data docker.elastic.co/elasticsearch/elasticsearch:8.15.2
}


function mk_es_dir() {
    sudo mkdir /es
    sudo chown -R 1000:0 /es
    sudo chown -R 777 /es
}