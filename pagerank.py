import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    pages = corpus.keys()
    num_pages = len(pages)
    links = corpus[page]

    if links:
        for p in pages:
            distribution[p] = (1 - damping_factor) / num_pages
        for link in links:
            distribution[link] += damping_factor / len(links)
    else:
        for p in pages:
            distribution[p] = 1 / num_pages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        page_rank[page] += 1
        distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(list(distribution.keys()), list(distribution.values()), k=1)[0]
    
    total_samples = sum(page_rank.values())
    page_rank = {page: rank / total_samples for page, rank in page_rank.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_page_rank = page_rank.copy()
    converged = False

    while not converged:
        converged = True
        for page in corpus:
            rank_sum = 0
            for p in corpus:
                if page in corpus[p]:
                    rank_sum += page_rank[p] / len(corpus[p])
                if len(corpus[p]) == 0:
                    rank_sum += page_rank[p] / num_pages
            new_page_rank[page] = (1 - damping_factor) / num_pages + damping_factor * rank_sum

        for page in page_rank:
            if abs(new_page_rank[page] - page_rank[page]) > 0.001:
                converged = False

        page_rank = new_page_rank.copy()

    return page_rank


if __name__ == "__main__":
    main()
