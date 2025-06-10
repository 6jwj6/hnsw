from Compiler.library import *
from Compiler.types import *
from Compiler.oram import *
# from Compiler.program import Program

ORAM = OptimalORAM

# try:
#     prog = program.Program.prog
#     prog.set_bit_length(min(64, prog.bit_length))
# except AttributeError:
#     pass

class HeapEntry(object):
    fields = ['empty', 'prio', 'value']
    def __init__(self, int_type, *args):
        self.int_type = int_type
        if not len(args):
            raise CompilerError()
        if len(args) == 1:
            args = args[0]
        for field,arg in zip(self.fields, args):
            self.__dict__[field] = arg
    def data(self):
        return self.prio, self.value
    def __repr__(self):
        return '(' + ', '.join('%s=%s' % (field,self.__dict__[field]) \
                                   for field in self.fields) + ')'
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __gt__(self, other):
        return (1 - self.empty) * (1 - other.empty) * \
            (self.int_type(self.prio) > self.int_type(other.prio))
    def __iter__(self):
        for field in self.fields:
            yield self.__dict__[field]
    def __add__(self, other):
        return type(self)(self.int_type, (i + j for i,j in zip(self, other)))
    def __sub__(self, other):
        return type(self)(self.int_type, (i - j for i,j in zip(self, other)))
    def __xor__(self, other):
        return type(self)(self.int_type, (i ^ j for i,j in zip(self, other)))
    def __mul__(self, other):
        return type(self)(self.int_type, (other * i for i in self))
    __rxor__ = __xor__
    __rmul__ = __mul__
    def hard_conv_me(self, value_type):
        return type(self)(self.int_type, \
                              *(value_type.hard_conv(x) for x in self))
    def dump(self):
        print_ln('empty %s, prio %s, value %s', *(reveal(x) for x in self))

class HeapORAM(object):
    def __init__(self, size, oram_type, init_rounds, int_type, entry_size=None):
        if entry_size is None:
            entry_size = (32,log2(size))
        self.int_type = int_type
        self.oram = oram_type(size, entry_size=entry_size, \
                                  init_rounds=init_rounds, \
                                  value_type=int_type.basic_type)
    def __getitem__(self, index):
        return self.make_entry(*self.oram.read(index))
    def make_entry(self, value, empty):
        return HeapEntry(self.int_type, (empty,) + value)
    def __setitem__(self, index, value):
        self.oram.access(index, value.data(), True, new_empty=value.empty)
    def access(self, index, value, write):
        tmp = self.oram.access(index, value.data(), write)
        return self.make_entry(*tmp)
    def delete(self, index, for_real):
        self.oram.delete(index, for_real)
    def read_and_maybe_remove(self, index):
        entry, state = self.oram.read_and_maybe_remove(index)
        return self.make_entry(*entry), state
    def add(self, index, entry, state):
        self.oram.add(Entry(MemValue(index), \
                                [MemValue(i) for i in entry.data()], \
                                entry.empty), state=state)
    def __len__(self):
        return len(self.oram)

class HeapQ(object):
    def __init__(self, max_size, oram_type=ORAM, init_rounds=-1, int_type=sint, entry_size=None):
        if entry_size is None:
            entry_size = (32, log2(max_size))
        basic_type = int_type.basic_type
        self.max_size = max_size
        self.levels = log2(max_size)
        self.depth = self.levels - 1
        self.heap = HeapORAM(2**self.levels, oram_type, init_rounds, int_type, entry_size=entry_size)
        self.value_index = oram_type(max_size, entry_size=entry_size[1], \
                                         init_rounds=init_rounds, \
                                         value_type=basic_type)
        self.size = MemValue(int_type(0))
        self.int_type = int_type
        self.basic_type = basic_type
        # prog.reading('heap queue', 'KS14')
        print('heap: %d levels, depth %d, size %d, index size %d' % \
            (self.levels, self.depth, self.heap.oram.size, self.value_index.size))
    
    def pop(self, for_real=True):
        '''
        pop and 输出 top 的 (prio, value)
        '''
        return self._pop(self.basic_type.hard_conv(for_real))
    def top(self, for_real=True):
        '''
        输出 top 的 (prio, value)
        '''
        return self._top(self.basic_type.hard_conv(for_real))
    def update(self, value, prio, for_real=True):
        '''
        更新 value 对应的 prio 值\n
        格式(value, prio)
        '''
        self._update(self.basic_type.hard_conv(value), \
                         self.basic_type.hard_conv(prio), \
                         self.basic_type.hard_conv(for_real))
    def bubble_up(self, start):
        bits = bit_decompose(start, self.levels)
        bits.reverse()
        bits = [0] + floatingpoint.PreOR(bits, self.levels)
        bits = [bits[i+1] - bits[i] for i in range(self.levels)]
        shift = self.int_type.bit_compose(bits)
        childpos = MemValue(start * shift)
        @for_range(self.levels - 1)
        def f(i):
            parentpos = childpos.right_shift(1, self.levels + 1)
            parent, parent_state = self.heap.read_and_maybe_remove(parentpos)
            child, child_state = self.heap.read_and_maybe_remove(childpos)
            swap = parent > child
            new_parent, new_child = cond_swap(swap, parent, child)
            self.heap.add(childpos, new_child, child_state)
            self.heap.add(parentpos, new_parent, parent_state)
            self.value_index.access(new_parent.value, parentpos, swap)
            self.value_index.access(new_child.value, childpos, swap)
            childpos.write(parentpos)
    @method_block
    def _top(self, for_real=True):
        '''
        输出 top\n 格式 (prio, value)
        '''
        # Program.prog.curr_tape.\
        #     start_new_basicblock(name='heapq-top')
        entry = self.heap[1]
        return (entry.prio, entry.value)
    @method_block
    def _pop(self, for_real=True):
        # Program.prog.curr_tape.\
        #     start_new_basicblock(name='heapq-pop')
        pop_for_real = for_real * (self.size != 0)
        entry = self.heap[1]
        self.value_index.delete(entry.value, for_real)
        last = self.heap[self.basic_type(self.size)]
        self.heap.access(1, last, pop_for_real)
        self.value_index.access(last.value, 1, for_real * (self.size != 1))
        self.heap.delete(self.basic_type(self.size), for_real)
        self.size -= self.int_type(pop_for_real)
        parentpos = MemValue(self.basic_type(1))
        @for_range(self.levels - 1)
        def f(i):
            childpos = 2 * parentpos
            left_child, l_state = self.heap.read_and_maybe_remove(childpos)
            right_child, r_state = self.heap.read_and_maybe_remove(childpos+1)
            go_right = left_child > right_child
            otherchildpos = childpos + 1 - go_right
            childpos += go_right
            child, other_child = cond_swap(go_right, left_child, right_child)
            child_state, other_state = cond_swap(go_right, l_state, r_state)
            parent, parent_state = self.heap.read_and_maybe_remove(parentpos)
            swap = parent > child
            new_parent, new_child = cond_swap(swap, parent, child)
            self.heap.add(childpos, new_child, child_state)
            self.heap.add(otherchildpos, other_child, other_state)
            self.heap.add(parentpos, new_parent, parent_state)
            self.value_index.access(new_parent.value, parentpos, swap)
            self.value_index.access(new_child.value, childpos, swap)
            parentpos.write(childpos)
        self.check()
        return (entry.prio, entry.value)
    @method_block
    def _update(self, value, prio, for_real=True):
        # Program.prog.curr_tape.\
        #     start_new_basicblock(name='heapq-update')
        index, not_found = self.value_index.read(value)
        self.size += self.int_type(not_found * for_real)
        index = if_else(not_found, self.basic_type(self.size), index[0])
        self.value_index.access(value, self.basic_type(self.size), \
                                    not_found * for_real)
        self.heap.access(index, HeapEntry(self.int_type, 0, prio, value), for_real)
        self.bubble_up(index)
        self.check()
    def __len__(self):
        return self.size
    def check(self):
        if debug:
            for i in range(len(self.heap)):
                if ((2 * i + 1 < len(self.heap) and \
                         self.heap[i] > self.heap[2*i+1]) or \
                        (2 * i + 2 < len(self.heap) and \
                             self.heap[i] > self.heap[2*i+2])) and \
                             not self.heap[i].empty:
                    raise Exception('heap condition violated at %d' % i)
                if i >= self.size and not self.heap[i].empty:
                    raise Exception('wrong size at %d' % i)
                if i < self.size and self.heap[i].empty:
                    raise Exception('empty entry in heap at %d' % i)
                # if not self.heap[i].empty and \
                #         self.heap[i].value not in self.value_index:
                #     raise Exception('missing index at %d' % i)
            for value,(index,empty) in enumerate(self.value_index):
                if not empty and self.heap[index].value != value:
                    raise Exception('index violated at %d' % index)
        if debug_online:
            @for_range(self.max_size)
            def f(value):
                index, not_found = self.value_index.read(value)
                index, not_found = index[0].reveal(), not_found.reveal()
                @if_(not_found == 0)
                def f():
                    heap_value = self.heap[index].value.reveal()
                    @if_(heap_value != value)
                    def f():
                        print_ln('heap mismatch: %s:%s in index, %s in heap', \
                                     value, index, heap_value)
                        crash()            
    def dump(self, msg=''):
        print_ln(msg)
        print_ln('size: %s', self.size.reveal())
        print_str('heap:')
        if isinstance(self.heap.oram, LinearORAM):
            for entry in self.heap.oram.ram:
                print_str(' %s:%s,%s', entry.empty().reveal(), \
                              entry.x[0].reveal(), entry.x[1].reveal())
        else:
            for i in range(self.max_size+1):
                print_str(' %s:%s', *(x.reveal() for x in self.heap.oram[i]))
        print_ln()
        print_str('value index:')
        if isinstance(self.value_index, LinearORAM):
            for entry in self.value_index.ram:
                print_str(' %s:%s', entry.empty().reveal(), entry.x[0].reveal())
        else:
            for i in range(self.max_size):
                print_str(' %s:%s', i, self.value_index[i].reveal())
        print_ln()
        print_ln()