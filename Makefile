WIKIDATA_FILE="wikidata.vec"
WIKIDATA_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"

# https://github.com/fennel-ai/fann/blob/main/src/benchmark.sh
download:
	if [ ! -f ${WIKIDATA_FILE} ]; then \
		echo "${WIKIDATA_FILE} does not exist. Downloading..."; \
		curl -o temporary.zip ${WIKIDATA_URL}; \
		unzip temporary.zip; \
		mv wiki-news-300d-1M.vec ${WIKIDATA_FILE}; \
		rm -rf temporary.zip; \
	else \
		echo "${WIKIDATA_FILE} already exists."; \
	fi
