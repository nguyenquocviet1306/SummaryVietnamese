import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
import nltk
from os.path import join as pjoin
from tensorflow.core.example import example_pb2

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',"& nbsp"]
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")
CHUNK_SIZE = 100

def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        try:
          str_len = struct.unpack('q', len_bytes)[0]
          example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, example_str))
        except:
          print("bo qua")
          continue
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train','val','test']:
    print ("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print ("Saved chunked data in %s" % chunks_dir)

def read_text_file(file_name):
  with open(file_name) as f:
    return [line.strip() for line in f]

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def get_art_abs(file_name):
  lines = read_text_file(file_name)
  # Lowercase everything
  lines = [line.lower() for line in lines]
  lines = [fix_missing_period(line) for line in lines]
  isAbstract, isHeader, isArticle = False, False, False
  abstracts = []
  articles = []
  headers = []
  curr_abstract = None
  curr_article = []
  curr_header = None
  for line in lines:
    if line.startswith("#"):
      i = 1
      if curr_abstract != None:
        abstracts.append(curr_abstract)
        headers.append(curr_header)
        articles.append(curr_article)
      isArticle = False
      curr_abstract = None
      curr_article = []
      curr_header = None
      isHeader = True
      continue
    if isHeader:
      curr_header = line
      isHeader = False
      isAbstract = True
    elif isAbstract:
      line2 = line.strip()
      if (line2[0] == "("):
        while(1):
          if(line2[0] != ")"):
            line2=line2.strip(line2[0])
          else:
            line2=line2.strip(line2[0])
            break
      line2_sent = line2.split()
      list_remove_word = ["ktđt","dân_trí","vov.vn","tp","tto","ĐTCK","PetroTimes","TBKTSG Online","DĐDN","Baodautu.vn","Tài_chính","VTC News","Xây_dựng","HNMO","HQ Online"]
      for word in list_remove_word:
        if(line2_sent[0] == word):
          line2 = line2.strip(line2_sent[0])                
      line2=line2.strip()
      if (line2[0] == "-"):
          line2 = line2.strip(line2[0])
      line2=line2.strip()
      curr_abstract =  '<s>'+ line2 + '</s>'
      isAbstract = False
      isArticle = True
    elif isArticle:
      sents = nltk.sent_tokenize(line)
      for sentence in sents: 
        # print(sentence)
        if sentence == "& nbsp ; .":
          continue
        curr_article.append('['+str(i)+'] '+sentence)
        i = i + 1 
  abstracts.append(curr_abstract)
  headers.append(curr_header)
  articles.append(curr_article)
  return abstracts, articles

def get_all_data(data_path):
  # all_headers, all_abstracts, all_articles = [], [], []
  all_abstracts, all_articles = [], []
  for file in os.listdir(data_path):
    try:
      # headers, abstracts, articles = get_art_abs(os.path.join(data_path, file))
      abstracts, articles = get_art_abs(os.path.join(data_path, file))
    except:
      print("bo qua file", file)
      continue
    # all_headers += headers
    all_abstracts += abstracts
    all_articles += articles
  return all_abstracts, all_articles
    
# def dump_into_binary(out_path, headers, abstracts, articles, makevocab=True):
def dump_into_binary(out_path, abstracts, articles, makevocab=True):
  num_stories = len(abstracts)
  if not os.path.exists(out_path):
    os.mkdir(out_path)
  # if makevocab:
  #   vocab_counter = collections.Counter()
  writer = open(os.path.join(out_path, "test.bin"), "wb")
  # for idx, (header, abstract, article) in enumerate(zip(headers, abstracts, articles)):
  for idx, (abstract, article) in enumerate(zip(abstracts, articles)):
    print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))
    tf_example = example_pb2.Example()
    # tf_example.features.feature['header'].bytes_list.value.extend([header.encode()])
    tf_example.features.feature['article'].bytes_list.value.extend([' '.join(article).strip().encode()])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str)) 
  
    # if makevocab:
    #   if article is not None:
    #     art_tokens = ' '.join(article).split(' ')
    #   # print(art_tokens)
    #   if abstract is not None:
    #     abs_tokens = abstract.split(' ')
    #     abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]]
    #   # print(abs_tokens)
    #   if header is not None:
    #     head_tokens = header.split(' ')
    #   # print(head_tokens)
    #   tokens = art_tokens + abs_tokens + head_tokens
    #   # print(tokens)
    #   tokens = [t.strip() for t in tokens] # strip
    #   # print("tokens-222")
    #   # print(tokens)
    #   tokens = [t for t in tokens if t!=""] # remove empty
    #   vocab_counter.update(tokens)

  # print("Finished writing file %s\n" % os.path.join(out_path, "output.bin"))

  # write vocab to file
  # if makevocab:
  #   print("Writing vocab file...")
  #   with open(os.path.join(out_path, "vocab"), 'w') as writer:
  #     for word, count in vocab_counter.items():
  #       writer.write(word + ' ' + str(count) + '\n')
  #   print("Finished writing vocab file")

if __name__ == '__main__':



  # where_to_look = '/home/vietnguyen/Documents/GR1/DataBaoMoi/BaoMoi/Vbee'
  # # print (len([f for f in os.listdir(where_to_look) if os.path.isfile(os.path.join(where_to_look, f))]))
  # for i in range(187):
  #   print (os.listdir(where_to_look)[i])
  #   filename = os.listdir(where_to_look)[i]
  #   path_source = pjoin(where_to_look,filename)
  #   print(path_source)
  #   path_to_file = pjoin("/home", "vietnguyen", "Documents", "GR1", "Resource", "textsum_vietnamese", "Tienxuly",
  #     "Vbee",filename)
  #   FILE = open(path_to_file, "w")
  #   command = ['java','-jar','uetsegmenter.jar', '-r', 'seg', '-m','/home/vietnguyen/Documents/GR1/Resource/UETsegmenter/models/',
  #     '-i',path_source,'-o',path_to_file]
  #   print(command)
  #   subprocess.call(command)
  # headers, abstracts, articles = get_all_data('./Test')
  abstracts, articles = get_all_data('./Test')
  # print(articles[1])
  # print(articles[1])

  dump_into_binary("finished_files",abstracts, articles)
  # chunk_all()



  # the_path = input("home/vietnguyen/Documents/GR1/DataBaoMoi/BaoMoi/BaoMoi")
  # # DIR = 'home/vietnguyen/Documents/GR1/DataBaoMoi/BaoMoi/BaoMoi'
  # filecount(the_path)
 # print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
  # if len(sys.argv) != 3:
  #   print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
  #   sys.exit()
  # cnn_stories_dir = sys.argv[1]
  # dm_stories_dir = sys.argv[2]

  # # Check the stories directories contain the correct number of .story files
  # check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  # check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # # Create some new directories
  # if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  # if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  # if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  # tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
  # tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

