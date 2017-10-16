"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode

# ====================
# Helper functions for manipulating bytes

def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int
    """
    return (byte & (1 << bit_num)) >> bit_num

def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])

def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])

# ====================
# Functions for compression

def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}
    """
    dct = {}
    for i in text:
        if i in dct:
            dct[i] += 1
        else:
            dct[i] = 1
    return dct

def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode
    """
    freq_copy = dict(freq_dict)
    lst = []
    while freq_copy:
        smallest = min(freq_copy, key=freq_copy.get)
        if freq_copy[smallest] > 0:
            lst.append((HuffmanNode(smallest), freq_copy[smallest]))
        del freq_copy[smallest]
    #Workaround for if there is only 1 char in a text file
    #0 is encoded as a null byte and should never appear except 
    #in an empty file, which isn't being tested.
    if len(lst) == 1:
        tree = HuffmanNode(None, lst[0][0], HuffmanNode(0))
        return tree
    while len(lst) > 1:
        flag = False
        small1 = lst.pop(0)
        small2 = lst.pop(0)
        freq1 = small1[1]
        freq2 = small2[1]
        freq_sum = freq1 + freq2
        temp = (HuffmanNode(None, small1[0], small2[0]), freq_sum) 
        for i in range(len(lst)):
            if freq_sum < lst[i][1] and not flag:
                lst.insert(i, temp)
                flag = True
        if not flag:
            lst.append(temp)
    return lst[0][0]

def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)
    """
    def helper1(tree, string):
        """Return a dictionary that maps all the symbols
        from the root of the tree to codes.
        
        @param1 HuffmanNode tree: a Huffman tree rooted at node 'tree'
        @param2 str string that represents the code that the tree will
        be converted to.
        @rtype: dict(int,str)
        """
        d = {}
        if not tree.left and not tree.right:
            d[tree.symbol] = string
        if tree.left:
            d.update(helper1(tree.left, string + "0"))
        if tree.right:
            d.update(helper1(tree.right, string + "1"))
        return d
    return helper1(tree, "")

def number_nodes(tree):
    """Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType
    """
    if tree.symbol is not None:
        tree.number = 0
        return
    def helper2(tree, num):
        """Number the internal nodes in the tree to poster order traversal, and
        return the number for internal nodes. Note: this return value is only 
        used inside the helper function.
        
        @param HuffmanNode tree, int; the number of nodes
        @rtype: int
        """
        if tree.left:
            num = helper2(tree.left, num)
        if tree.right:
            num = helper2(tree.right, num)
        if tree.symbol is None:
            tree.number = num
            num += 1
        return num
    helper2(tree, 0)
    return

def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float
    """
    codes = get_codes(tree)
    lst = []
    sum_freq = 0
    for key in freq_dict:
        lst.append(len(codes[key]) * freq_dict[key])
        sum_freq += freq_dict[key]
    return sum(lst)/sum_freq

def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes
    """
    text = list(text)
    bits = ""
    for i in text:
        bits += codes[i]
    while len(bits) % 8 != 0:
        bits += "0"
    index = 8
    lst = []
    while index <= len(bits) + 1:
        temp = bits[index - 8: index]
        lst.append(bits_to_byte(temp))
        index += 8
    return bytes(lst)
    
def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.
    """
    def helper3(tree):
        """Return a list representation of the tree rooted at tree.
        
        The representation is then simply converted to bytes by tree_to_bytes.
        Precondition: tree has its nodes numbered.
        
        @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
        @rtype: list
        """
        lst = []
        if tree.left:
            lst += helper3(tree.left)
        if tree.right:
            lst += helper3(tree.right)
        if tree.number is not None and tree.left is not None:
            if tree.left.number is None:
                lst.append(0)
                lst.append(tree.left.symbol)
            elif tree.left.number is not None:
                lst.append(1)
                lst.append(tree.left.number)
            if tree.right.number is None:
                lst.append(0)
                lst.append(tree.right.symbol)
            elif tree.right.number is not None:
                lst.append(1)
                lst.append(tree.right.number)
        return lst
    lst = helper3(tree)
    return bytes(lst)
    
def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])

def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")

def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)

# ====================
# Functions for decompression

def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode
    """
    tree = HuffmanNode(None, None, None)
    def helper4(node_lst, root_index, tree):
        """Helps modify the inputted tree, by adding Nodes from the
        given node_lst, and root_index.
        
        @param lst[ReadNode] node_lst a list of ReadNode objects
        @param root_index, int; the root of the tree
        @param tree, HuffmanNode tree
        
        @rtype: NoneType
        """
        l_data = node_lst[root_index].l_data
        r_data = node_lst[root_index].r_data
        if node_lst[root_index].l_type == 0:
            tree.left = HuffmanNode(l_data)
        else:
            tree.left = HuffmanNode(None, None, None)
            helper4(node_lst, l_data, tree.left)
        if node_lst[root_index].r_type == 0:
            tree.right = HuffmanNode(r_data)
        else:
            tree.right = HuffmanNode(None, None, None)
            helper4(node_lst, r_data, tree.right)
        return
    helper4(node_lst, root_index, tree)
    return tree
        
def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode
    """
    tree = HuffmanNode(None, None, None)
    def helper5(node_lst, root_index, tree):
        """Modify tree to create a Huffman tree using node_lst and root_index
        Return a temporary index value used to retrieve values from node_lst.
        Note the returned value is unused outside of the helper function.
        
        @param list[ReadNode] node_lst: a list of ReadNode objects
        @param int root_index: index in the node list
        @rtype: int
        """
        l_data = node_lst[root_index].l_data
        r_data = node_lst[root_index].r_data
        
        if node_lst[root_index].l_type == 0:
            tree.left = HuffmanNode(l_data) 
        if node_lst[root_index].r_type == 0:
            tree.right = HuffmanNode(r_data)
        temp_index = root_index    
        if node_lst[root_index].r_type == 1:
            temp_index -= 1
            tree.right = HuffmanNode(None, None, None)
            temp_index = helper5(node_lst, temp_index, tree.right)

        if node_lst[root_index].l_type == 1:
            temp_index -= 1 
            tree.left = HuffmanNode(None, None, None)
            temp_index = helper5(node_lst, temp_index, tree.left)
        return temp_index
    
    helper5(node_lst, root_index, tree)
    return tree

def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    bits = ''
    for i in text:
        bits += byte_to_bits(i)
    lst = []
    tree_copy = tree
    i = 0
    while len(lst) < size:
        if bits[i] == "0":
            tree_copy = tree_copy.left
        elif bits[i] == "1":
            tree_copy = tree_copy.right
        if tree_copy.symbol is not None:
            lst.append(tree_copy.symbol)
            tree_copy = tree 
        i += 1
    return bytes(lst)

def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst

def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int
    """
    return int.from_bytes(buf, "little")

def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))

# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType
    """
    #Sorted List of the symbols of High to Low frequency.
    lst = sorted(freq_dict, key=freq_dict.__getitem__)[::-1]
    new_tree = [tree]
    
    while lst != []:
        new_tree1 = []
        
        for i in new_tree:
            if not i.right.left and not i.right.right:
                i.right.symbol = lst[0]
                lst.remove(i.right.symbol)
            
            else:
                new_tree1.append(i.right)
            
            if not i.left.left and not i.left.right:
                i.left.symbol = lst[0]
                lst.remove(i.left.symbol)
                
            else:
                new_tree1.append(i.left)
            
        new_tree = new_tree1[:]
        new_tree1 = []

if __name__ == "__main__":
    
    import time
    
    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start)) 