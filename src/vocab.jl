import Base: length

SOS = "<sos>" # start of sentence token
EOS = "<eos>" # end of sentence token
UNK = "<unk>" # token for unknown words
PAD = "<pad>" # padding token

mutable struct Vocabulary
    counts # word counts dict
    sorted # sorted word counts tuple, for stats
    w2i # word to index dict
    i2w # index to word array
    min_occur # minimum occurence

    function Vocabulary(words::Array{Any,1}, min_occur=5)
        # get word counts
        counts = Dict()
        for word in words
            if haskey(counts, word)
                counts[word] += 1
            else
                counts[word] = 1
            end
        end

        # filter less occured words, build word2index dict upon that collection
        counts = filter(p->p[2] >= min_occur , counts)
        sorted = sort(collect(counts), by = tuple -> last(tuple), rev=true)
        w2i = Dict()

        i = 1
        for (w,o) in sorted
            w2i[w] = i
            i += 1
        end

        w2i[SOS] = i
        w2i[EOS] = i+1
        w2i[UNK] = i+2
        w2i[PAD] = i+3

        # build index2word array
        vocabsize = length(values(w2i))
        i2w = map(j -> "", zeros(vocabsize))
        for (k,v) in w2i
            i2w[v] = k
        end

        new(counts, sorted, w2i, i2w, min_occur)
    end
end


word2index(voc::Vocabulary, w) = haskey(voc.w2i, w) ? voc.w2i[w] : voc.w2i[UNK]
index2word(voc::Vocabulary, i) = voc.i2w[i]
most_occurs(voc::Vocabulary, N) = map(x -> (x.first, y.first), voc.sorted[1:N])
vec2sen(voc::Vocabulary, vec) = join(map(i -> index2word(voc,i), vec[2:end-1]), " ")
length(voc::Vocabulary) = length(voc.w2i)
