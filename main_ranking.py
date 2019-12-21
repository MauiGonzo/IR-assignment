""""
    main_ranking.py

    main file for the IR assignment. with the help of arguments you able/enable a certain modus
    build:      means you want to build the top n mini corpi
    rank_BM25:  rank according tot the rank BM25Okapi library
    rank_NN:    rank accoriding to the MatchZoo library with the NN sorting algorithm
"""
import argparse
import ranking_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ranker', nargs='+', type=str,
                        help='pick <build, rank_BM25, rank_NN>')
    args = parser.parse_args()

    base_path = '/Users/mauriceverbrugge/Google Drive/COMPUTING SCIENCE/IR/IR_assignment/robust04/'
    for rnk in args.ranker:
        if rnk == 'build':
            print('Building mini-corpi:')
            ranking_utils.init_minicorpi(base_path, 1000)

            #do_stuff
        elif rnk == 'rank_BM25':
            print(f'Rank with: {rnk}')
            # do stuff
        elif rnk == 'rank_NN':
            print(f'Rank with: {rnk}')
            # do stuff

