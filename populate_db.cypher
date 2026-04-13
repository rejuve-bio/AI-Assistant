// 1. Create Genes
CREATE (g1:gene {
  id: 'ensg00000012048',
  gene_name: 'BRCA1',
  gene_type: 'protein_coding',
  chr: 'chr17',
  start: '43044295',
  end: '43170245'
});
CREATE (g2:gene {
  id: 'ensg00000139618',
  gene_name: 'BRCA2',
  gene_type: 'protein_coding',
  chr: 'chr13',
  start: '32315086',
  end: '32400268'
});
CREATE (g3:gene {
  id: 'ensg00000141510',
  gene_name: 'TP53',
  gene_type: 'protein_coding',
  chr: 'chr17',
  start: '7661779',
  end: '7687550'
});

// 2. Create Transcripts
CREATE (t1:transcript {
  id: 'enst00000357654',
  transcript_name: 'BRCA1-201',
  transcript_id: 'ENST00000357654',
  gene_name: 'BRCA1',
  chr: 'chr17',
  start: '43044295',
  end: '43170245'
});
CREATE (t2:transcript {
  id: 'enst00000380152',
  transcript_name: 'BRCA2-201',
  transcript_id: 'ENST00000380152',
  gene_name: 'BRCA2',
  chr: 'chr13',
  start: '32315086',
  end: '32400268'
});
CREATE (t3:transcript {
  id: 'enst00000269305',
  transcript_name: 'TP53-201',
  transcript_id: 'ENST00000269305',
  gene_name: 'TP53',
  chr: 'chr17',
  start: '7661779',
  end: '7687550'
});

// 3. Create Exons
CREATE (e1:exon {
  id: 'ense00003509998',
  exon_id: 'ENSE00003509998',
  exon_number: '1',
  chr: 'chr17',
  start: '43044295',
  end: '43045802',
  transcript_id: 'ENST00000357654',
  gene_id: 'ENSG00000012048'
});
CREATE (e2:exon {
  id: 'ense00001893883',
  exon_id: 'ENSE00001893883',
  exon_number: '2',
  chr: 'chr17',
  start: '43047643',
  end: '43047703',
  transcript_id: 'ENST00000357654',
  gene_id: 'ENSG00000012048'
});

// 4. Create Proteins
CREATE (p1:protein {
  id: 'p38398',
  protein_name: 'BRCA1',
  accessions: '["P38398"]'
});
CREATE (p2:protein {
  id: 'p51587',
  protein_name: 'BRCA2',
  accessions: '["P51587"]'
});
CREATE (p3:protein {
  id: 'p04637',
  protein_name: 'TP53',
  accessions: '["P04637"]'
});

// 5. Create Relationships
MATCH (g:gene {gene_name: 'BRCA1'}), (t:transcript {transcript_name: 'BRCA1-201'})
CREATE (g)-[:transcribed_to]->(t);
MATCH (t:transcript {transcript_name: 'BRCA1-201'}), (g:gene {gene_name: 'BRCA1'})
CREATE (t)-[:transcribed_from]->(g);
MATCH (t:transcript {transcript_name: 'BRCA1-201'}), (e:exon {exon_id: 'ENSE00003509998'})
CREATE (t)-[:includes]->(e);
MATCH (t:transcript {transcript_name: 'BRCA1-201'}), (e:exon {exon_id: 'ENSE00001893883'})
CREATE (t)-[:includes]->(e);
MATCH (t:transcript {transcript_name: 'BRCA1-201'}), (p:protein {protein_name: 'BRCA1'})
CREATE (t)-[:translates_to]->(p);
MATCH (p:protein {protein_name: 'BRCA1'}), (t:transcript {transcript_name: 'BRCA1-201'})
CREATE (p)-[:translation_of]->(t);

// Repeat for BRCA2
MATCH (g:gene {gene_name: 'BRCA2'}), (t:transcript {transcript_name: 'BRCA2-201'})
CREATE (g)-[:transcribed_to]->(t);
MATCH (t:transcript {transcript_name: 'BRCA2-201'}), (g:gene {gene_name: 'BRCA2'})
CREATE (t)-[:transcribed_from]->(g);
MATCH (t:transcript {transcript_name: 'BRCA2-201'}), (p:protein {protein_name: 'BRCA2'})
CREATE (t)-[:translates_to]->(p);

// Repeat for TP53
MATCH (g:gene {gene_name: 'TP53'}), (t:transcript {transcript_name: 'TP53-201'})
CREATE (g)-[:transcribed_to]->(t);
MATCH (t:transcript {transcript_name: 'TP53-201'}), (g:gene {gene_name: 'TP53'})
CREATE (t)-[:transcribed_from]->(g);
MATCH (t:transcript {transcript_name: 'TP53-201'}), (p:protein {protein_name: 'TP53'})
CREATE (t)-[:translates_to]->(p);
