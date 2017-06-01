import pysolr

solr = pysolr.Solr('http://localhost:8983/solr/sparksolr/', timeout=10)
results = solr.search('*:*', **{"fl": "field*"})
print list(results)