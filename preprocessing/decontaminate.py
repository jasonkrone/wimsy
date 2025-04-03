import os
import sys
import pdb
import glob
import json
import logging
import traceback
from datetime import datetime, timezone
from collections import defaultdict
from multiprocessing import Pool

import smart_open
import pandas as pd
from tqdm import tqdm
from jsonpath_ng import parse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from tokenizers.pre_tokenizers import CharDelimiterSplit

from preprocessing.constants import NgramConstants, SkypilotConstants
from utils import Registry, logger, timeit, get_value_at_jsonpath, ParallelTQDM

logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


def get_metadata_subdir_name():
    rank = SkypilotConstants.get_rank(default=0) 
    world_size = SkypilotConstants.get_world_size(default=1)
    return f"node-{rank+1}-of-{world_size}"


@Registry.register("decontaminator")
class Decontaminator(object):

    def __init__(
        self,
        metadata_prefix,
        drop_doc_contamination_threshold, 
        replacement_text, 
        drop_entire_doc, 
        is_dryrun, 
        num_processes,
        property_to_jsonpath,
        raise_error_on_id_mismatch=True,
    ):
        self.drop_doc_contamination_threshold = drop_doc_contamination_threshold
        self.replacement_text = replacement_text
        self.drop_entire_doc = drop_entire_doc
        self.is_dryrun = is_dryrun
        self.num_processes = num_processes

        self.metadata_dir = os.path.join(metadata_prefix, get_metadata_subdir_name())
        self.aggregate_stats_save_path = os.path.join(self.metadata_dir, "avg_decontam_stats.jsonl")
        self.per_doc_stats_save_path = os.path.join(self.metadata_dir, "per_doc_decontam_stats.jsonl")
        self.contaminated_spans_path = os.path.join(self.metadata_dir, "contaminated_spans.jsonl")

        self.property_to_jsonpath = {property: parse(jsonpath) for property, jsonpath in property_to_jsonpath.items()}
        self.raise_error_on_id_mismatch = raise_error_on_id_mismatch
        assert "id" in self.property_to_jsonpath and "text" in self.property_to_jsonpath

    def __call__(self):
        if not os.path.exists(self.contaminated_spans_path):
            logger.warning(f"Contaminated span file not present at: {self.contaminated_spans_path}")
            return

        num_docs_total = 0
        num_docs_dropped = 0
        stats_df_list = []

        contam_df = self._df_for_jsonl(self.contaminated_spans_path)
        num_contaminated_docs = len(contam_df)

        path_to_location_to_spans = defaultdict(lambda: defaultdict(list))
        for _, row in contam_df.iterrows():
            location = (int(row["line_num"]), row["id"])
            path_to_location_to_spans[row["path"]][location] = row["spans"]

        tqdm_parallel = ParallelTQDM()
        pbar_desc = "Replacing contaminated spans in files"
        pbar_queue = tqdm_parallel.start_pbar(num_contaminated_docs, pbar_desc)

        num_processes = min(len(path_to_location_to_spans), self.num_processes)
        args_list = [
            (
                path,
                location_to_spans, 
                self.replacement_text, 
                self.drop_entire_doc, 
                self.drop_doc_contamination_threshold, 
                self.is_dryrun, 
                pbar_queue,
                self.property_to_jsonpath,
                self.raise_error_on_id_mismatch,
            )
            for path, location_to_spans in path_to_location_to_spans.items()
        ]

        with Pool(processes=num_processes) as pool:
            for i, (stats_list, stats_dict) in enumerate(pool.starmap(self.decontaminate_file, args_list)):
                stats_df_list += stats_list
                num_docs_dropped += stats_dict["num_docs_dropped"]
                num_docs_total += stats_dict["num_docs_total"]
                # update metadata doc count for file
                metadata_path = f"{args_list[i][0]}.done.txt"
                with smart_open.open(metadata_path, "wt") as f:
                    f.write(json.dumps({"num_documents": stats_dict["num_docs_total"]}))

        tqdm_parallel.stop_pbar()

        os.makedirs(os.path.dirname(self.per_doc_stats_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.aggregate_stats_save_path), exist_ok=True)

        stats_df = pd.DataFrame(stats_df_list)
        stats_df.to_json(self.per_doc_stats_save_path, orient="records", lines=True)

        aggregate_stats = {
            "num_docs_dropped": num_docs_dropped, 
            "num_docs_total": num_docs_total, 
            "percent_docs_dropped": num_docs_dropped / float(num_docs_total) * 100,
        }
        with open(self.aggregate_stats_save_path, "w") as f:
            json.dump(aggregate_stats, f)
        logger.info(f"Decontaminator dropped {aggregate_stats['percent_docs_dropped']} percent of docs")

    @classmethod
    def _pbar_listener(cls, q, total):
        """
        Written by GPT-4    
        """
        pbar = tqdm(total=total, desc="Replacing contaminated spans in files", mininterval=30)
        for item in iter(q.get, None):
            pbar.update(item)
        pbar.close()

    @classmethod
    def decontaminate_file(
        cls, 
        path, 
        location_to_spans, 
        replacement_text, 
        drop_entire_doc, 
        drop_doc_contamination_threshold, 
        is_dryrun, 
        pbar_queue,
        property_to_jsonpath,
        raise_error_on_id_mismatch,
        pbar_interval=500,
    ):
        delta_i = 0
        line_num = 0
        num_dropped = 0
        stats_df_list = []
        temp_path = os.path.join(os.path.dirname(path), f"temp_{os.path.basename(path)}")
        lines_and_ids = sorted(location_to_spans.keys(), key=lambda t: t[0])
    
        with smart_open.open(path) as f, smart_open.open(temp_path, "wt") as f_temp:
            while True:
                line = f.readline()
                if line:
                    json_dict = json.loads(line)
                    contam_line_num, contam_doc_id = lines_and_ids[0] if len(lines_and_ids) else (None, None)
                    doc_id = get_value_at_jsonpath(json_dict, property_to_jsonpath["id"])

                    if lines_and_ids and line_num == contam_line_num and contam_doc_id == doc_id:
                        # we don't store percent contaminated if drop_entire_doc is True
                        percent_contam = None
                        key = lines_and_ids.pop(0)
                        do_drop_doc = drop_entire_doc
                        # check if % of chars contaminated is over threshold
                        if not drop_entire_doc:
                            text = get_value_at_jsonpath(json_dict, property_to_jsonpath["text"])
                            contam_spans = location_to_spans[key]
                            num_chars_in_doc = len(text)
                            num_contam_chars = sum([end - start for start, end in contam_spans])
                            percent_contam = num_contam_chars / float(num_chars_in_doc)
                            do_drop_doc = percent_contam >= drop_doc_contamination_threshold

                        # if doc is kept, replace contaminated span
                        if not do_drop_doc:
                            property_to_jsonpath["text"].update(json_dict, cls.replace_spans(text, contam_spans, replacement_text))
                            json.dump(json_dict, f_temp, separators=(",", ":"))
                            f_temp.write("\n")

                        # update the pbar
                        delta_i += 1
                        if delta_i % pbar_interval == 0 or len(lines_and_ids) == 0:
                            pbar_queue.put(delta_i)
                            delta_i = 0

                        # update stats
                        num_dropped += int(do_drop_doc)
                        stats_df_list.append({"path": path, "line_num": line_num, "dropped": do_drop_doc, "percent_contaminated": percent_contam, "success": True})
                    elif lines_and_ids and line_num == contam_line_num and contam_doc_id != doc_id:
                        lines_and_ids.pop(0)
                        stats_df_list.append({"path": path, "line_num": line_num, "dropped": False, "percent_contaminated": None, "success": False})
                        f_temp.write(line)
                        message = f"doc at path: {path} line number: {line_num} has ID {doc_id} != contam ID {contam_doc_id}. No edit to doc made."
                        if raise_error_on_id_mismatch:
                            raise RuntimeError(message)
                        else:
                            logger.error(message)
                    else:
                        f_temp.write(line)
                    line_num += 1
                else:
                    break

        if not is_dryrun:
            os.replace(temp_path, path)

        summary_stats_dict = {
            "num_docs_total": line_num,
            "num_docs_dropped": num_dropped,
        }

        return stats_df_list, summary_stats_dict

    @classmethod
    def replace_spans(cls, text, spans, replacement):
        """
        Replaces specified spans of text in the input text with the given replacement text.

        Written by GPT-4
        """
        # Sort the spans in reverse order to avoid index shifting
        spans = sorted(spans, reverse=True)
        for start, end in spans:
            text = text[:start] + replacement + text[end:]
        return text

    def _df_for_jsonl(self, path):
        json_list = []
        with smart_open.open(path) as f:
            lines = f.readlines()
            for l in lines:
                json_list.append(json.loads(l))
        return pd.DataFrame(json_list)


class ElasticSearchPreprocessingStep(object):

    def __init__(self, index_name, hosts, request_timeout, max_retries, num_processes, pbar_interval=500):

        self.num_processes = num_processes
        self.index_name = index_name
        self.pbar_interval = pbar_interval

        self.es = Elasticsearch(
            hosts=hosts,
            connections_per_node=self.num_processes,
            request_timeout=request_timeout, 
            http_compress=True,
            max_retries=max_retries,
            retry_on_timeout=True,
        )

    def _safe_save_as_jsonl(self, df_list, save_path, message):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
 
        if len(df_list) > 0:
            index_failures_df = pd.DataFrame(df_list)
            index_failures_df.to_json(save_path, lines=True, orient="records")
            logger.info(message)



@Registry.register("contaminated_span_tagger")
class ContaminatedSpanTagger(ElasticSearchPreprocessingStep):

    MAX_HITS = 10_000
    MAX_HITS_TIMEOUT = 3600 # 1hr
    MAX_HITS_PIT_KEEP_ALIVE = "60m"
    MAX_TOKEN_COUNT = 100_000_000

    def __init__(
        self, 
        ngram_queries_path, 
        ngram_sources_path,
        search_after_field,
        ckpt_interval, 
        metadata_prefix,
        do_find_contaminated_spans,
        micro_batch_size,
        num_ngrams=None, 
        round_to_delimiter=None, 
        ckpt_path=None,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.num_ngrams = num_ngrams
        self.ngram_queries_path = ngram_queries_path 
        self.ngram_sources_path = ngram_sources_path
        self.do_find_contaminated_spans = do_find_contaminated_spans
        self.round_to_delimiter = round_to_delimiter
        self.search_after_field = search_after_field
        self.micro_batch_size = micro_batch_size

        self.metadata_dir = os.path.join(metadata_prefix, get_metadata_subdir_name())
        self.contaminated_spans_save_path = os.path.join(self.metadata_dir, "contaminated_spans.jsonl")
        self.max_hits_save_path = os.path.join(self.metadata_dir, "queries_with_max_hits.jsonl")
        self.contaminated_sources_save_path = os.path.join(self.metadata_dir, "contaminated_sources.jsonl")

        self.ckpt_interval = ckpt_interval
        self.doc_to_contaminated_spans = defaultdict(list)
        self.source_to_contaminated_ngrams = defaultdict(list)
        with smart_open.open(self.ngram_sources_path, "rt") as f:
            self.ngram_to_sources = json.load(f)

        self.ckpt_iter = 0
        if ckpt_path:
            self.ckpt_iter, self.doc_to_contaminated_spans, self.source_to_contaminated_ngrams = self._load_ckpt(ckpt_path)

    def __call__(self, **kwargs):
        prev_ckpt_iter = self.ckpt_iter 
        queries_with_max_hits = []
        ngram_gen = self.read_ngrams(self.ngram_queries_path, self.ckpt_iter)
        ngram_batches = self.make_batches(ngram_gen, batch_size=self.num_processes, micro_batch_size=self.micro_batch_size)
        pbar = tqdm(total=self.num_ngrams, initial=self.ckpt_iter, desc="running decontamination queries", mininterval=30)

        self._set_max_tokens(self.MAX_TOKEN_COUNT)

        for batch in ngram_batches:
            # get all the documents that contain any of the ngrams in the batch
            result_list = self._get_documents_containing_phrases_parallel(
                es=self.es, index=self.index_name, phrase_batch=batch, max_concurrent_searches=self.num_processes,
            )
            # extract contaminated spans from results
            for results, ngram_list in zip(result_list, batch):
                # more than max results, re-run search to get all matching docs
                if len(results) >= self.MAX_HITS:
                    queries_with_max_hits.append({"ngrams": ngram_list})
                    results = self._get_all_documents_containing_phrases(
                        es=self.es, index=self.index_name, phrases=ngram_list, search_after_field=self.search_after_field
                    )

                for hit in results:
                    # TODO: depending on how slow the analyze API is we could write a tokenizer that does the same thing
                    # tokenize source document
                    doc_text = hit["_source"]["text"]
                    doc_tokens = self.es.indices.analyze(index=self.index_name, body={"analyzer": "standard", "text": doc_text})["tokens"]

                    # tag the source document as contaminated
                    contaminated_spans = []
                    if self.do_find_contaminated_spans:
                        contaminated_spans = self._get_spans_of_ngrams_in_text(
                            doc_text, doc_tokens, ngram_list, is_greedy=False, round_to_delimiter=self.round_to_delimiter
                        )
                    key = (hit["_source"]["id"], hit["_source"]["path"], hit["_source"]["line_num"])
                    self.doc_to_contaminated_spans[key] += contaminated_spans

                    for ngram in self._get_contaminated_ngrams(ngram_list, doc_tokens):
                        for source_dict in self.ngram_to_sources[ngram]:
                            key = (source_dict["source"], source_dict["split"], source_dict["index"], source_dict["num_ngrams"])
                            self.source_to_contaminated_ngrams[key].append(ngram)

                pbar.update(len(ngram_list))

            if pbar.n - prev_ckpt_iter >= self.ckpt_interval:
                dir_path, filename = os.path.split(self.contaminated_spans_save_path)
                basename, _ = os.path.splitext(filename)
                ckpt_path = os.path.join(dir_path, f"ckpt_{basename}.json") 
                self._write_ckpt(pbar.n, self.doc_to_contaminated_spans, self.source_to_contaminated_ngrams, ckpt_path)
                prev_ckpt_iter = pbar.n

        # mark the task as complete. needed in case the est num ngrams is wrong
        if pbar.total and pbar.n < pbar.total:
            pbar.total = pbar.n
            pbar.refresh()

        contaminated_spans_df_list = self._span_dict_to_df_list(self.doc_to_contaminated_spans)
        self._safe_save_as_jsonl(
            contaminated_spans_df_list,
            self.contaminated_spans_save_path,
            f"Wrote the {len(contaminated_spans_df_list)} contaminated spans to {self.contaminated_spans_save_path}"
        )

        contaminated_sources_df_list = self._source_dict_to_df_list(self.source_to_contaminated_ngrams)
        self._safe_save_as_jsonl(
            contaminated_sources_df_list,
            self.contaminated_sources_save_path,
            f"Wrote the {len(contaminated_sources_df_list)} contaminated sources to {self.contaminated_sources_save_path}"
        )

        self._safe_save_as_jsonl(
            queries_with_max_hits, 
            self.max_hits_save_path, 
            f"Wrote the {len(queries_with_max_hits)} queries with max hits to {self.max_hits_save_path}"
        )

    @classmethod
    def _load_ckpt(cls, save_path):
        with smart_open.open(save_path, "rt") as f:
            ckpt_dict = json.loads(f.read())

        doc_to_contaminated_spans = defaultdict(list)
        for row in ckpt_dict.pop("contaminated_spans"):
            key = (row["id"], row["path"], row["line_num"])
            # tuples get cast as lists when we write to ckpt
            doc_to_contaminated_spans[key] += [tuple(span) for span in row["spans"]]

        source_to_contaminated_ngrams = defaultdict(list)
        for row in ckpt_dict.pop("contaminated_sources"):
            key = (row["source"], row["split"], row["index"], row["num_source_ngrams"])
            source_to_contaminated_ngrams[key] = row["contaminated_ngrams"]

        return ckpt_dict["iter"], doc_to_contaminated_spans, source_to_contaminated_ngrams

    @classmethod
    def _write_ckpt(cls, iter, contaminated_spans_dict, contaminated_sources_dict, save_path):
        contaminated_sources_df_list = cls._source_dict_to_df_list(contaminated_sources_dict)
        contaminated_spans_df_list = cls._span_dict_to_df_list(contaminated_spans_dict)
        ckpt_dict = {
            "iter": iter, 
            "contaminated_spans": contaminated_spans_df_list,
            "contaminated_sources": contaminated_sources_df_list,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with smart_open.open(save_path, "wt") as f:
            f.write(json.dumps(ckpt_dict))

    @classmethod
    def _get_documents_containing_phrases_parallel(cls, es, index, phrase_batch, max_concurrent_searches):
        """
        Inspired by https://github.com/allenai/wimbd/blob/72695d60c349c02d96899207ada9fabf7e4d0c4b/wimbd/es/__init__.py
        """
        msearch_body = []
        for phrases in phrase_batch:
            assert isinstance(phrases, list)
            msearch_body.append({"index": index})
            # find docs where at least one of the phrases is present
            query = cls._get_query_to_match_phrases(phrases)
            msearch_body.append({
                "size": cls.MAX_HITS,
                "query": query,
            })
        responses = es.msearch(body=msearch_body, max_concurrent_searches=max_concurrent_searches)
        hits = [response["hits"]["hits"] for response in responses["responses"]]
        return hits

    @classmethod
    def _get_all_documents_containing_phrases(cls, es, index, phrases, search_after_field):
        """
        Search which allows to retrieval of all matching results via search after.
        Inspired by https://github.com/allenai/wimbd/blob/72695d60c349c02d96899207ada9fabf7e4d0c4b/wimbd/es/__init__.py
        """
        query = cls._get_query_to_match_phrases(phrases)
        sort = [{search_after_field: "asc"}]
        pit = es.open_point_in_time(index=index, keep_alive=cls.MAX_HITS_PIT_KEEP_ALIVE)
        pit_id = pit["id"]
        pit_dict = {"id": pit_id, "keep_alive": cls.MAX_HITS_PIT_KEEP_ALIVE}
        try:
            hits = es.search(
                query=query, size=cls.MAX_HITS, sort=sort, pit=pit_dict, 
                request_timeout=cls.MAX_HITS_TIMEOUT,
            )["hits"]["hits"]
            yield from hits
            while hits:
                hits = es.search(
                    query=query, size=cls.MAX_HITS, sort=sort, search_after=hits[-1]["sort"], 
                    pit=pit_dict, request_timeout=cls.MAX_HITS_TIMEOUT,
                )["hits"]["hits"]
                yield from hits
        finally:
            es.close_point_in_time(body={"id": pit_id})

    @classmethod
    def _get_query_to_match_phrases(cls, phrases):
        query = {
            "bool": {
                "should": [{"match_phrase": {"text": {"query": q, "slop": 0}}} for q in phrases], 
                "minimum_should_match": 1
            }
        }
        return query

    # TODO: test this
    @classmethod
    def _get_contaminated_ngrams(cls, query_ngrams, source_tokens):
        source_str = NgramConstants.TOKEN_DELIMITER.join([t["token"] for t in source_tokens])
        contam_ngrams = [ngram for ngram in query_ngrams if ngram in source_str]
        return contam_ngrams

    @classmethod
    def _get_spans_of_ngrams_in_text(cls, source_text, source_tokens, contam_ngram_list, is_greedy, round_to_delimiter):
        """
        Returns the spans of any contam text within source text. If is_greedy is set, matches as many characters
        are possible when there is an ambiguity. If round_to_delimiter is not None, rounds the spans to the nearest
        delimiter boundaries. All contaminated tokens must be the same length e.g., all n-grams of the same size.
        """
        span_list = []
        n_source_tokens = len(source_tokens)
        contam_ngram_list = [ngram.split(" ") for ngram in contam_ngram_list]
        ngram_len = len(contam_ngram_list[0])

        for i in range(n_source_tokens-ngram_len+1):
            source_ngram = [t["token"] for t in source_tokens[i:i+ngram_len]]
            is_match = any([contam_ngram == source_ngram for contam_ngram in contam_ngram_list])
            
            if is_match:
                span_start = source_tokens[i]["start_offset"]
                span_end = source_tokens[i+ngram_len-1]["end_offset"]
                
                if is_greedy:
                    # check if there's a next token
                    if i + ngram_len < n_source_tokens:
                        span_end = source_tokens[i+ngram_len]["start_offset"]
                    # check if there's a prior token    
                    if i - 1 >= 0:
                        span_start = source_tokens[i-1]["end_offset"]
                    
                span_list.append((span_start, span_end))

        if round_to_delimiter:
            pre_tokenizer = CharDelimiterSplit(round_to_delimiter)
            delimiter_spans = [span for _, span in pre_tokenizer.pre_tokenize_str(source_text)]
            span_list = cls._round_spans_to_nearest_delimiter(span_list, delimiter_spans)

        span_list = cls._merge_contiguous_spans(span_list)
        return span_list

    @classmethod
    def _merge_contiguous_spans(cls, spans):
        """
        Written by GPT o1-preview
        """
        if not spans:
            return []

        merged_spans = [spans[0]]
        for current_start, current_end in spans[1:]:
            last_start, last_end = merged_spans[-1]
            # Check if spans are contiguous or overlapping
            if current_start <= last_end + 1:
                # Merge the spans
                merged_spans[-1] = (last_start, max(last_end, current_end))
            else:
                merged_spans.append((current_start, current_end))
        return merged_spans

    @classmethod
    def _round_spans_to_nearest_delimiter(cls, spans, delimiter_spans, is_greedy=True):
        """
        Rounds the given spans to the span of the nearest delimiter. 
        If is_greedy == True, includes the trailing delimiter in the span. 
        """
        rounded_spans = []
        assert len(delimiter_spans) > 0
        
        # we can deactivate delimiters as span start is less than 
        active_delimiters = delimiter_spans
        
        for start, end in spans:
            # remove delimiter spans that won't match 
            _, delimiter_end = active_delimiters[0]
            # delimiter ends before span starts
            while delimiter_end <= start:
                active_delimiters.pop(0)
                _, delimiter_end = active_delimiters[0]

            # Find the delimiter with the largest start <= position
            for i, (delimiter_start, delimiter_end) in enumerate(active_delimiters):
                if delimiter_start <= start:
                    start = delimiter_start
                if delimiter_end >= end:
                    # there's a next delimiter span
                    if is_greedy and i < len(active_delimiters) - 1:
                        end = active_delimiters[i+1][0]
                    else:
                        end = delimiter_end
                    # once we've found the end, we will have already found the start
                    break
                    
            rounded_spans.append((start, end))

        return rounded_spans

    @classmethod
    def _source_dict_to_df_list(cls, source_to_ngrams):
        df_list = []
        for (source, split, index, num_source_ngrams), contam_ngrams in source_to_ngrams.items():
            df_list.append({
                "source": source, 
                "split": split, 
                "index": index, 
                "num_source_ngrams": num_source_ngrams, 
                "contaminated_ngrams": contam_ngrams
            })
        return df_list

    @classmethod
    def _span_dict_to_df_list(cls, doc_id_to_spans):
        df_list = []
        for (doc_id, doc_path, line_num), spans in tqdm(doc_id_to_spans.items(), desc="post-processing contaminated spans"):
            spans = list(set(spans))
            spans = sorted(spans, key=lambda t: t[0])
            spans = cls._merge_contiguous_spans(spans)
            df_list.append({"id": doc_id, "path": doc_path, "line_num": line_num, "spans": spans})
        return df_list
        
    def _set_max_tokens(self, max_token_count):
        new_settings = {"index": {"analyze": {"max_token_count": max_token_count}}}
        self.es.indices.put_settings(index=self.index_name, body=new_settings)
    
    @classmethod
    def read_ngrams(cls, path, ckpt_iter, ngram_column="ngram_for_decontam"):
        """
        Reads ngrams from the path starting with the ngram at ckpt_iter + 1.
        """
        num_skipped = 0
        with smart_open.open(path) as f:
            while num_skipped < ckpt_iter:
                line = f.readline()
                json_dict = json.loads(line)
                num_skipped += 1
            assert num_skipped == ckpt_iter

            while True:
                line = f.readline()
                if line:
                    json_dict = json.loads(line)
                    yield json_dict[ngram_column]
                else:
                    break

    @classmethod
    def make_batches(cls, gen, batch_size, micro_batch_size):
        batch = []
        micro_batch = []
        for item in gen:
            micro_batch.append(item)
            if len(micro_batch) == micro_batch_size:
                batch.append(micro_batch)
                micro_batch = []
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        # Handle any remaining items in the last micro batch
        if micro_batch:
            batch.append(micro_batch)
        # Handle any remaining micro batches in the last batch
        if batch:
            yield batch


@Registry.register("elastic_search_snapshotter")
class ElasticSearchSnapshotter(ElasticSearchPreprocessingStep):

    # 5hr
    SNAPSHOT_TIMEOUT = 18_000

    def __init__(self, repository_name, repository_settings, **kwargs):
        super().__init__(**kwargs)
        self.repository_name = repository_name
        self.repository_settings = repository_settings
        # update repository name and settings to factor fo rrank
        rank = SkypilotConstants.get_rank(default=0)
        world_size = SkypilotConstants.get_world_size(default=1)
        rank_str = f"rank_{rank}_of_world_size_{world_size}"
        self.repository_name = f"{self.index_name}_{rank_str}"
        self.repository_settings["settings"]["base_path"] = os.path.join(self.index_name, rank_str)

    @timeit(fn_name="elastic_search_snapshotter")
    def __call__(self, **kwargs):
        self.es.snapshot.create_repository(
            name=self.repository_name,
            body=self.repository_settings,
            verify=True
        )

        snapshot_name = f"{self.repository_name}_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        response = self.es.snapshot.create(
            repository=self.repository_name,
            snapshot=snapshot_name,
            body={
                "indices": self.index_name,
                "ignore_unavailable": False,
                "include_global_state": False,
            },
            wait_for_completion=True,
            request_timeout=self.SNAPSHOT_TIMEOUT, 
        )

        failures = response["snapshot"]["failures"]
        if len(failures) > 0:
            raise ValueError(f"failed to create snapshot {snapshot_name}. Failures: {failures}")
        else:
            logger.info(f"created snapshot: {snapshot_name} with settings: {self.repository_settings}")


@Registry.register("elastic_search_indexer")
class ElasticSearchIndexer(ElasticSearchPreprocessingStep):

    def __init__(
        self, 
        source_glob, 
        metadata_prefix,
        properties, 
        property_to_jsonpath, 
        chunk_size, 
        num_documents_to_index=None, 
        raise_on_index_failure=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source_paths = glob.glob(source_glob)

        self.metadata_dir = os.path.join(metadata_prefix, get_metadata_subdir_name())
        self.failures_save_path = os.path.join(self.metadata_dir, "indexing_failures.jsonl")
        self.stats_save_path = os.path.join(self.metadata_dir, "indexing_stats.jsonl")

        self.chunk_size = chunk_size
        self.num_documents_to_index = num_documents_to_index
        self.raise_on_index_failure = raise_on_index_failure

        self.property_to_jsonpath = {
            property: parse(jsonpath) for property, jsonpath in property_to_jsonpath.items()
        }
        self.properties = properties

        # make the index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self.create_index(self.index_name, self.properties)

    def create_index(self, index_name, properties):
        self.es.indices.create(index=index_name, mappings={"properties": properties})

    @timeit(fn_name="elastic_search_indexer")
    def __call__(self, **kwargs):
        n_missing_docs = 0
        n_docs_in_index = 0
        index_stats_df_list = []
        index_failures_df_list = []
        n_files_to_index = len(self.source_paths)
        assert n_files_to_index
        pbar = tqdm(total=self.num_documents_to_index, desc="indexing files to elastic search", mininterval=30)

        for i, path in enumerate(self.source_paths):
            prev_n_lines = pbar.n
            n_docs_added_to_index = 0

            try:
                row_generator = self.get_rows_to_index(path, self.index_name, pbar)
                out = parallel_bulk(
                    client=self.es,
                    actions=row_generator,
                    thread_count=self.num_processes,
                    chunk_size=self.chunk_size,
                    raise_on_exception=False,
                    queue_size=self.num_processes,
                    refresh="wait_for",
                )

                for is_success, info_dict in out:
                    n_docs_added_to_index += int(is_success)
                    if not is_success:
                        logger.error(f"indexing file {i+1} with path {path} failed (at least partially). error info: {info_dict}")
                        index_failures_df_list.append({"path": path, "info": info_dict})

            except Exception as e:
                error_str = traceback.format_exc()
                logger.error(f"indexing file {i+1} with path {path} failed with error: {error_str}")
                index_failures_df_list.append({"path": path, "info": {"error": error_str}})

            n_lines = pbar.n - prev_n_lines
            delta = n_lines - n_docs_added_to_index
            n_docs_in_index += n_docs_added_to_index
            n_missing_docs += delta

            # estimate total based on number of lines in the first doc
            if pbar.total is None:
                pbar.total = n_lines * n_files_to_index
                pbar.refresh()

            if n_lines == n_docs_added_to_index:
                logger.info(f"Indexed file {i+1} of {n_files_to_index}.")
            else:
                message = f"Partially indexed file {i+1} of {n_files_to_index} with path {path}. Read {n_lines} lines, added {n_docs_added_to_index} rows. Delta: {delta}"
                if self.raise_on_index_failure:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)

            index_stats_df_list.append({"path": path, "num_lines": n_lines, "num_docs_added_to_index": n_docs_added_to_index, "delta": delta})

        # mark the task as complete. needed in case the est num docs is wrong
        if pbar.total and pbar.n < pbar.total:
            pbar.total = pbar.n
            pbar.refresh()

        logger.info(f"Finished! Indexed {n_docs_in_index} documents total.")
        self._safe_save_as_jsonl(index_failures_df_list, self.failures_save_path, f"Wrote index failures to: {self.failures_save_path}")
        self._safe_save_as_jsonl(index_stats_df_list, self.stats_save_path, f"Wrote index stats to: {self.stats_save_path}")
        if n_missing_docs:
            percent_missing_docs = n_missing_docs / float(pbar.n) * 100
            logger.warning(f"Missing {n_missing_docs} docs ({percent_missing_docs} percent)")

    def get_rows_to_index(self, path, index_name, pbar):
        line_num = 0
        with smart_open.open(path) as f:
            while True:
                line = f.readline()
                if line:                    
                    json_dict = json.loads(line)
                    source_dict = {
                        property: get_value_at_jsonpath(json_dict, jsonpath)
                        for property, jsonpath in self.property_to_jsonpath.items()
                    }
                    if "created" not in source_dict or source_dict["created"] == "":
                        now_utc = datetime.now(timezone.utc)
                        source_dict["created"] = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                    source_dict["path"] = path
                    source_dict["line_num"] = line_num
                    yield {"_index": index_name, "_source": source_dict}
                    pbar.update(1) 
                    line_num += 1
                else:
                    break

        metadata_path = f"{path}.done.txt"
        with smart_open.open(metadata_path, "wt") as f:
            f.write(json.dumps({"num_documents": line_num}))


