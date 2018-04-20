import parlai.core.build_data as build_data
import os
import pickle

def create_fb_format(data, dpath, subreddit=None):
    train_file = 'train_' + subreddit + '.txt' if subreddit else 'train.txt'
    test_file = 'test_' + subreddit + '.txt' if subreddit else 'text.txt'
    valid_file = 'valid_' + subreddit + '.txt' if subreddit else 'valid.txt'
    fw1 = open(os.path.join(dpath, train_file), 'w')
    fw2 = open(os.path.join(dpath, valid_file), 'w')
    fw3 = open(os.path.join(dpath, test_file), 'w')
    i = 0
    for link_id in data:
        for dialog in data[link_id]:
            fout = fw1
            if (i % 500) == 0:
                fout = fw2
            elif (i % 500) == 1:
                fout = fw3
            i += 1
            num = 1
            for j in range(0, len(dialog) - 1, 2): # if odd messages, the last one will be omitted
                x = dialog[j]['body'].rstrip(' ').lstrip(' ').replace('\n', ' ').replace('\t', ' ')
                y = dialog[j + 1]['body'].rstrip(' ').lstrip(' ').replace('\n', ' ').replace('\t', ' ')
                s = str(num) + ' ' + x + '\t' + y
                fout.write(s + '\n')
                num += 1

    fw1.close()
    fw2.close()
    fw3.close()

def create_fb_format_by_link(data, dpath, subtask=None):
    train_file = 'train_' + subtask + '.txt' if subtask else 'train.txt'
    test_file = 'test_' + subtask + '.txt' if subtask else 'text.txt'
    valid_file = 'valid_' + subtask + '.txt' if subtask else 'valid.txt'
    fw1 = open(os.path.join(dpath, train_file), 'w')
    fw2 = open(os.path.join(dpath, valid_file), 'w')
    fw3 = open(os.path.join(dpath, test_file), 'w')

    if subtask.endswith('_QA'):
        QA = True
    else:
        QA = False

    i = 0
    for link_id in data:
        fout = fw1
        if (i % 150) == 0:
            fout = fw2
        elif (i % 150) == 1:
            fout = fw3
        for dialog in data[link_id]:
            i += 1
            num = 1
            for j in range(0, len(dialog) - 1, 2): # if odd messages, the last one will be omitted
                x = dialog[j]['body'].rstrip(' ').lstrip(' ').replace('\n', ' ').replace('\t', ' ')
                y = dialog[j + 1]['body'].rstrip(' ').lstrip(' ').replace('\n', ' ').replace('\t', ' ')
                s = str(num) + ' ' + x + '\t' + y
                fout.write(s + '\n')
                if not QA: # for dialogue, we make the beginning number grow to indicate the turn of dialogues, but for QA we don't.
                    num += 1

    fw1.close()
    fw2.close()
    fw3.close()
    
def build(opt, subtask=None):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'Reddit', subtask)

    # check if data had been previously built
    if not build_data.built(dpath, version_string=subtask):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # don't download the data.
        fname = os.environ['HOME'] + '/data/anime.pickle'
        if subtask:
            fname = os.environ['HOME'] + '/data/' + subtask + '.pickle'
        data = pickle.load(open(fname, 'rb'))

        # create_fb_format(data, dpath, subtask)
        create_fb_format_by_link(data, dpath, subtask)

        # mark the data as built
        build_data.mark_done(dpath, version_string=subtask)
