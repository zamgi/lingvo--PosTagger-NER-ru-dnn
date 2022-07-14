using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Threading;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace System.Collections.Generic
{
    //
    // A binary search tree is a red-black tree if it satisfies the following red-black properties:
    // 1. Every node is either red or black
    // 2. Every leaf (nil node) is black
    // 3. If a node is red, the both its children are black
    // 4. Every simple path from a node to a descendant leaf contains the same number of black nodes
    // 
    // The basic idea of red-black tree is to represent 2-3-4 trees as standard BSTs but to add one extra bit of information  
    // per node to encode 3-nodes and 4-nodes. 
    // 4-nodes will be represented as:          B
    //                                                              R            R
    // 3 -node will be represented as:           B             or         B     
    //                                                              R          B               B       R
    // 
    // For a detailed description of the algorithm, take a look at "Algorithm" by Rebert Sedgewick.
    //

    [SuppressMessage("Microsoft.Naming", "CA1710:IdentifiersShouldHaveCorrectSuffix", Justification="by design name choice")]
    [DebuggerTypeProxy( typeof(ICollectionDebugView<>) )]
    [DebuggerDisplay("Count = {Count}")]
    public sealed class SortedSetByRef< T > : ISet< T >, ICollection< T >, ICollection, IReadOnlyCollection< T > where T : struct
    {
        #region [.ctor().]
        private const int STACK_ALLOC_THRESHOLD = 100;

        private Node _Root;
        //---private object _SyncRoot;

        [M(O.AggressiveInlining)] public SortedSetByRef( IComparerByRef< T > comparer ) => Comparer = comparer ?? throw (new ArgumentNullException( nameof(comparer) )); 
        public SortedSetByRef( IComparerByRef< T > comparer, IEnumerable< T > collection ) : this( comparer )
        {
            if ( collection == null ) throw (new ArgumentNullException( nameof(collection) ));

            // these are explicit type checks in the mould of HashSet. It would have worked better
            // with something like an ISorted< T > (we could make this work for SortedList.Keys etc)
            var baseSortedSet  = collection as SortedSetByRef< T >;
            if ( (baseSortedSet != null) && AreComparersEqual( this, baseSortedSet ) )
            {
                //breadth first traversal to recreate nodes
                if ( baseSortedSet.Count == 0 )
                {
                    return;
                }

                //pre order way to replicate nodes
                var capacity     = 2 * log2( baseSortedSet.Count ) + 2;
                var theirStack   = new Stack< Node >( capacity );
                var myStack      = new Stack< Node >( capacity );
                var theirCurrent = baseSortedSet._Root;
                var myCurrent    = ((theirCurrent != null) ? new Node( theirCurrent.Item, theirCurrent.IsRed ) : null);
                _Root = myCurrent;
                while ( theirCurrent != null )
                {
                    theirStack.Push( theirCurrent );
                    myStack.Push( myCurrent );
                    myCurrent.Left = (theirCurrent.Left != null ? new Node( theirCurrent.Left.Item, theirCurrent.Left.IsRed ) : null);
                    theirCurrent = theirCurrent.Left;
                    myCurrent = myCurrent.Left;
                }
                while ( theirStack.Count != 0 )
                {
                    theirCurrent = theirStack.Pop();
                    myCurrent = myStack.Pop();
                    var theirRight = theirCurrent.Right;
                    Node myRight = null;
                    if ( theirRight != null )
                    {
                        myRight = new Node( theirRight.Item, theirRight.IsRed );
                    }
                    myCurrent.Right = myRight;

                    while ( theirRight != null )
                    {
                        theirStack.Push( theirRight );
                        myStack.Push( myRight );
                        myRight.Left = (theirRight.Left != null ? new Node( theirRight.Left.Item, theirRight.Left.IsRed ) : null);
                        theirRight = theirRight.Left;
                        myRight = myRight.Left;
                    }
                }
                Count = baseSortedSet.Count;
            }
            else
            {
                T[] els = EnumerableHelpers.ToArray( collection, out var count );
                if ( 0 < count )
                {
                    throw (new NotImplementedException());

                    /*
                    comparer = Comparer; // If comparer is null, sets it to Comparer< T >.Default
                    Array.Sort( els, 0, count, comparer );
                    var index = 1;
                    for ( var i = 1; i < count; i++ )
                    {
                        if ( comparer.Compare( els[ i ], els[ i - 1 ] ) != 0 )
                        {
                            els[ index++ ] = els[ i ];
                        }
                    }
                    count = index;

                    _Root = ConstructRootFromSortedArray( els, 0, count - 1, null );
                    Count = count;
                    //*/
                }
            }
        }
        #endregion

        #region [.Bulk Operation Helpers.]
        [M(O.AggressiveInlining)] private void AddAllElements( IEnumerable< T > collection )
        {
            foreach ( var item in collection )
            {
                if ( !Contains( in item ) )
                {
                    Add( in item );
                }
            }
        }
        [M(O.AggressiveInlining)] private void RemoveAllElements( IEnumerable< T > collection )
        {
            T min = Min;
            T max = Max;
            foreach ( var item in collection )
            {
                if ( !(Comparer.Compare( in item, in min ) < 0 || Comparer.Compare( in item, in max ) > 0) && Contains( in item ) )
                {
                    Remove( in item );
                }
            }
        }
        [M(O.AggressiveInlining)] private bool ContainsAllElements( IEnumerable< T > collection )
        {
            foreach ( var item in collection )
            {
                if ( !Contains( in item ) )
                {
                    return (false);
                }
            }
            return (true);
        }

        // Do a in order walk on tree and calls the delegate for each node.
        // If the action delegate returns false, stop the walk.
        // 
        // Return true if the entire tree has been walked. 
        // Otherwise returns false.
        private bool InOrderTreeWalk( TreeWalkPredicate< T > action ) => InOrderTreeWalk( action, false );

        // Allows for the change in traversal direction. Reverse visits nodes in descending order 
        private bool InOrderTreeWalk( TreeWalkPredicate< T > action, bool reverse )
        {
            if ( _Root == null )
            {
                return (true);
            }

            // The maximum height of a red-black tree is 2*lg(n+1).
            // See page 264 of "Introduction to algorithms" by Thomas H. Cormen
            // note: this should be logbase2, but since the stack grows itself, we 
            // don't want the extra cost
            var stack = new Stack< Node >( 2 * log2( Count + 1 ) );
            var current = _Root;
            while ( current != null )
            {
                stack.Push( current );
                current = (reverse ? current.Right : current.Left);
            }
            while ( stack.Count != 0 )
            {
                current = stack.Pop();
                if ( !action( current ) )
                {
                    return (false);
                }

                var node = (reverse ? current.Left : current.Right);
                while ( node != null )
                {
                    stack.Push( node );
                    node = (reverse ? node.Right : node.Left);
                }
            }
            return (true);
        }

        // Do a left to right breadth first walk on tree and 
        // calls the delegate for each node.
        // If the action delegate returns false, stop the walk.
        // 
        // Return true if the entire tree has been walked. 
        // Otherwise returns false.
        private bool BreadthFirstTreeWalk( TreeWalkPredicate< T > action )
        {
            if ( _Root == null )
            {
                return (true);
            }

            var processQueue = new Queue< Node >();
            processQueue.Enqueue( _Root );

            while ( processQueue.Count != 0 )
            {
                var current = processQueue.Dequeue();
                if ( !action( current ) )
                {
                    return (false);
                }
                if ( current.Left != null )
                {
                    processQueue.Enqueue( current.Left );
                }
                if ( current.Right != null )
                {
                    processQueue.Enqueue( current.Right );
                }
            }
            return (true);
        }
        #endregion

        #region [.Props.]
        public int Count { [M(O.AggressiveInlining)] get; private set; }
        public IComparerByRef< T > Comparer { [M(O.AggressiveInlining)] get; }
        #endregion

        #region [.methods.]
        /// <summary>
        /// Add the value ITEM to the tree, returns true if added, false if duplicate 
        /// </summary>
        /// <param name="item">item to be added</param> 
        public bool Add( in T item ) => AddIfNotPresent( in item );
        /// <summary>
        /// Adds ITEM to the tree if not already present. Returns TRUE if value was successfully added or FALSE if it is a duplicate
        /// </summary>        
        private bool AddIfNotPresent( in T item )
        {
            if ( _Root == null )
            {   // empty tree
                _Root = new Node( in item, false );
                Count = 1;
                /*_version++;*/
                return (true);
            }

            // Search for a node at bottom to insert the new node. 
            // If we can guarantee the node we found is not a 4-node, it would be easy to do insertion.
            // We split 4-nodes along the search path.
            Node current = _Root;
            Node parent           = null;
            Node grandParent      = null;
            Node greatGrandParent = null;

            //even if we don't actually add to the set, we may be altering its structure (by doing rotations
            //and such). so update version to disable any enumerators/subsets working on it
            /*_version++;*/

            int order = 0;
            while ( current != null )
            {
                order = Comparer.Compare( in item, in current.Item );
                if ( order == 0 )
                {
                    // We could have changed root node to red during the search process.
                    // We need to set it to black before we return.
                    _Root.IsRed = false;
                    return (false);
                }

                // split a 4-node into two 2-nodes                
                if ( Is4Node( current ) )
                {
                    Split4Node( current );
                    // We could have introduced two consecutive red nodes after split. Fix that by rotation.
                    if ( IsRed( parent ) )
                    {
                        InsertionBalance( current, ref parent, grandParent, greatGrandParent );
                    }
                }
                greatGrandParent = grandParent;
                grandParent = parent;
                parent = current;
                current = (order < 0) ? current.Left : current.Right;
            }

#if DEBUG
            Debug.Assert( parent != null, "Parent node cannot be null here!" ); 
#endif
            // ready to insert the new node
            var node = new Node( in item );
            if ( 0 < order )
            {
                parent.Right = node;
            }
            else
            {
                parent.Left = node;
            }

            // the new node will be red, so we will need to adjust the colors if parent node is also red
            if ( parent.IsRed )
            {
                InsertionBalance( node, ref parent, grandParent, greatGrandParent );
            }

            // Root node is always black
            _Root.IsRed = false;
            ++Count;
            return (true);
        }

        /// <summary>
        /// Remove the T ITEM from this SortedSet. Returns true if successfully removed.
        /// </summary>
        public bool Remove( in T item )
        {
            if ( _Root == null )
            {
                return (false);
            }

            // Search for a node and then find its successor. 
            // Then copy the item from the successor to the matching node and delete the successor. 
            // If a node doesn't have a successor, we can replace it with its left child (if not empty.) 
            // or delete the matching node.
            // 
            // In top-down implementation, it is important to make sure the node to be deleted is not a 2-node.
            // Following code will make sure the node on the path is not a 2 Node. 

            //even if we don't actually remove from the set, we may be altering its structure (by doing rotations
            //and such). so update version to disable any enumerators/subsets working on it
            /*_version++;*/

            Node current       = _Root;
            Node parent        = null;
            Node grandParent   = null;
            Node match         = null;
            Node parentOfMatch = null;
            bool foundMatch    = false;
            while ( current != null )
            {
                if ( Is2Node( current ) )
                { // fix up 2-Node
                    if ( parent == null )
                    {   // current is root. Mark it as red
                        current.IsRed = true;
                    }
                    else
                    {
                        Node sibling = GetSibling( current, parent );
                        if ( sibling.IsRed )
                        {
                            // If parent is a 3-node, flip the orientation of the red link. 
                            // We can achieve this by a single rotation        
                            // This case is converted to one of other cased below.
#if DEBUG
                            Debug.Assert( !parent.IsRed, "parent must be a black node!" ); 
#endif
                            if ( parent.Right == sibling )
                            {
                                RotateLeft( parent );
                            }
                            else
                            {
                                RotateRight( parent );
                            }

                            parent.IsRed = true;
                            sibling.IsRed = false;    // parent's color
                            // sibling becomes child of grandParent or root after rotation. Update link from grandParent or root
                            ReplaceChildOfNodeOrRoot( grandParent, parent, sibling );
                            // sibling will become grandParent of current node 
                            grandParent = sibling;
                            if ( parent == match )
                            {
                                parentOfMatch = sibling;
                            }

                            // update sibling, this is necessary for following processing
                            sibling = (parent.Left == current) ? parent.Right : parent.Left;
                        }
#if DEBUG
                        Debug.Assert( sibling != null && !sibling.IsRed, "sibling must not be null and it must be black!" ); 
#endif
                        if ( Is2Node( sibling ) )
                        {
                            Merge2Nodes( parent, current, sibling );
                        }
                        else
                        {
                            // current is a 2-node and sibling is either a 3-node or a 4-node.
                            // We can change the color of current to red by some rotation.
                            TreeRotation rotation = RotationNeeded( parent, current, sibling );
                            Node newGrandParent = null;
                            switch ( rotation )
                            {
                                case TreeRotation.RightRotation:
#if DEBUG
                                    Debug.Assert( parent.Left == sibling, "sibling must be left child of parent!" );
                                    Debug.Assert( sibling.Left.IsRed, "Left child of sibling must be red!" );
#endif                                
                                    sibling.Left.IsRed = false;
                                    newGrandParent = RotateRight( parent );
                                break;

                                case TreeRotation.LeftRotation:
#if DEBUG
                                    Debug.Assert( parent.Right == sibling, "sibling must be left child of parent!" );
                                    Debug.Assert( sibling.Right.IsRed, "Right child of sibling must be red!" ); 
#endif
                                    sibling.Right.IsRed = false;
                                    newGrandParent = RotateLeft( parent );
                                break;

                                case TreeRotation.RightLeftRotation:
#if DEBUG
                                    Debug.Assert( parent.Right == sibling, "sibling must be left child of parent!" );
                                    Debug.Assert( sibling.Left.IsRed, "Left child of sibling must be red!" ); 
#endif
                                    newGrandParent = RotateRightLeft( parent );
                                break;

                                case TreeRotation.LeftRightRotation:
#if DEBUG
                                    Debug.Assert( parent.Left == sibling, "sibling must be left child of parent!" );
                                    Debug.Assert( sibling.Right.IsRed, "Right child of sibling must be red!" ); 
#endif
                                    newGrandParent = RotateLeftRight( parent );
                                break;
                            }

                            newGrandParent.IsRed = parent.IsRed;
                            parent.IsRed  = false;
                            current.IsRed = true;
                            ReplaceChildOfNodeOrRoot( grandParent, parent, newGrandParent );
                            if ( parent == match )
                            {
                                parentOfMatch = newGrandParent;
                            }
                            grandParent = newGrandParent;
                        }
                    }
                }

                // we don't need to compare any more once we found the match
                var order = (foundMatch ? -1 : Comparer.Compare( in item, in current.Item ));
                if ( order == 0 )
                {
                    // save the matching node
                    foundMatch = true;
                    match = current;
                    parentOfMatch = parent;
                }

                grandParent = parent;
                parent = current;

                if ( order < 0 )
                {
                    current = current.Left;
                }
                else
                {
                    current = current.Right;       // continue the search in  right sub tree after we find a match
                }
            }

            // move successor to the matching node position and replace links
            if ( match != null )
            {
                ReplaceNode( match, parentOfMatch, parent, grandParent );
                --Count;
            }

            if ( _Root != null )
            {
                _Root.IsRed = false;
            }
            return (foundMatch);
        }

        public void Clear()
        {
            _Root = null;
            Count = 0;
        }
        public bool Contains( in T item ) => (FindNode( in item ) != null);
        public bool TryGetValue( in T item, out T existsItem )
        {
            var node = FindNode( in item );
            if ( node != null )
            {
                existsItem = node.Item;
                return (true);
            }
            existsItem = default;
            return (false);
        }
        #endregion

        #region [.ICollection< T > Members.]
        bool ICollection< T >.IsReadOnly => false;
        bool ICollection.IsSynchronized => false;
        object ICollection.SyncRoot => throw (new NotImplementedException());
        //{
        //    get
        //    {
        //        if ( _SyncRoot == null )
        //        {
        //            Interlocked.CompareExchange( ref _SyncRoot, new object(), null );
        //        }
        //        return (_SyncRoot);
        //    }
        //}

        void ICollection< T >.Add( T item ) => AddIfNotPresent( in item );
        bool ICollection< T >.Contains( T item ) => Contains( in item );
        bool ICollection< T >.Remove( T item ) => Remove( in item );

        void ICollection.CopyTo( Array array, int index )
        {
            if ( array == null )                  throw (new ArgumentNullException( nameof(array) ));
            if ( array.Rank != 1 )                throw (new ArgumentException( "SR.Arg_RankMultiDimNotSupported", nameof(array) ));
            if ( array.GetLowerBound( 0 ) != 0 )  throw (new ArgumentException( "SR.Arg_NonZeroLowerBound", nameof(array) ));
            if ( index < 0 )                      throw (new ArgumentOutOfRangeException( nameof(index), index, "SR.ArgumentOutOfRange_NeedNonNegNum" ));
            if ( (array.Length - index) < Count ) throw (new ArgumentException( "SR.Arg_ArrayPlusOffTooSmall" ));

            var tarray = array as T[];
            if ( tarray != null )
            {
                CopyTo( tarray, index, Count );
            }
            else
            {
                var objects = array as object[];
                if ( objects == null ) throw (new ArgumentException( "SR.Argument_InvalidArrayType", nameof(array) ));

                try
                {
                    InOrderTreeWalk( (n) =>
                    { 
                        objects[ index++ ] = n.Item; 
                        return (true); 
                    });
                }
                catch ( ArrayTypeMismatchException )
                {
                    throw (new ArgumentException( "SR.Argument_InvalidArrayType", nameof(array) ));
                }
            }
        }
        void ICollection< T >.CopyTo( T[] array, int index ) => CopyTo( array, index, Count );

        public void CopyTo( T[] array, int index, int count )
        {
            if ( array == null ) throw (new ArgumentNullException( nameof(array) ));
            if ( index < 0 )     throw (new ArgumentOutOfRangeException( nameof(index), index, "SR.ArgumentOutOfRange_NeedNonNegNum" ));
            if ( count < 0 )     throw (new ArgumentOutOfRangeException( nameof(count), "SR.ArgumentOutOfRange_NeedNonNegNum" ));

            // will array, starting at arrayIndex, be able to hold elements? Note: not
            // checking arrayIndex >= array.Length (consistency with list of allowing
            // count of 0; subsequent check takes care of the rest)
            if ( (array.Length < index) || ((array.Length - index) < count) ) throw (new ArgumentException( "SR.Arg_ArrayPlusOffTooSmall" ));
            
            //upper bound
            count += index;

            InOrderTreeWalk( (node) => 
            {
                if ( index < count )
                {
                    array[ index++ ] = node.Item;
                    return (true);                    
                }

                return (false);
            });
        }
        public void CopyTo( T[] array ) => CopyTo( array, 0, Count );
        #endregion

        #region [.IEnumerable< T > members.]
        IEnumerator< T > IEnumerable< T >.GetEnumerator() => new Enumerator( this );
        public Enumerator GetEnumerator() => new Enumerator( this );
        IEnumerator IEnumerable.GetEnumerator() => new Enumerator( this );

        /// <summary>
        /// 
        /// </summary>
        [SuppressMessage("Microsoft.Performance", "CA1815:OverrideEqualsAndOperatorEqualsOnValueTypes", Justification="not an expected scenario")]
        public struct Enumerator : IEnumerator< T >, IEnumerator
        {
            private SortedSetByRef< T > _Tree;
            private Stack< Node >  _Stack;
            private Node _Current;

            [M(O.AggressiveInlining)] internal Enumerator( SortedSetByRef< T > set )
            {
                _Tree = set;

                // 2lg(n + 1) is the maximum height
                _Stack   = new Stack< Node >( log2( set.Count + 1 ) << 1 );
                _Current = null;

                Intialize();
            }

            [M(O.AggressiveInlining)] private void Intialize()
            {
                _Current = null;
                var node = _Tree._Root;
                while ( node != null )
                {
                    _Stack.Push( node );
                    node = node.Left; //---(_Reverse ? node.Right : node.Left);
                }
            }

            [M(O.AggressiveInlining)] public bool MoveNext()
            {
                if ( _Stack.Count == 0 )
                {
                    _Current = null;
                    return (false);
                }

                _Current = _Stack.Pop();
                var node = _Current.Right; //---(_Reverse ? _Current.Left : _Current.Right);
                while ( node != null )
                {
                    _Stack.Push( node );
                    node = node.Left; //---(_Reverse ? node.Right : node.Left);
                }
                return (true);
            }
            public void Dispose() { }

            public T Current
            {
                [M(O.AggressiveInlining)] get
                {
                    if ( _Current != null )
                    {
                        return (_Current.Item);
                    }
                    return (default);
                }
            }
            object IEnumerator.Current
            {
                get
                {
                    if ( _Current == null ) throw (new InvalidOperationException( "SR.InvalidOperation_EnumOpCantHappen" ));
                    return (_Current.Item);
                }
            }

            internal void Reset()
            {
                _Stack.Clear();
                Intialize();
            }
            void IEnumerator.Reset() => Reset();
        }

        /// <summary>
        /// 
        /// </summary>
        public struct Enumerator_4_Reverse : IEnumerator< T >, IEnumerator
        {
            private SortedSetByRef< T > _Tree;
            private Stack< Node >  _Stack;
            private Node _Current;

            internal Enumerator_4_Reverse( SortedSetByRef< T > set )
            {
                _Tree = set;

                // 2lg(n + 1) is the maximum height
                _Stack   = new Stack< Node >( 2 * log2( set.Count + 1 ) );
                _Current = null;

                Intialize_Reverse();
            }

            [M(O.AggressiveInlining)] private void Intialize_Reverse()
            {
                _Current = null;
                var node = _Tree._Root;
                while ( node != null )
                {
                    _Stack.Push( node );
                    node = node.Right; //---(_Reverse ? node.Right : node.Left);
                }
            }

            [M(O.AggressiveInlining)] public bool MoveNext()
            {
                if ( _Stack.Count == 0 )
                {
                    _Current = null;
                    return (false);
                }

                _Current = _Stack.Pop();
                var node = _Current.Left; //---(_Reverse ? _Current.Left : _Current.Right);
                while ( node != null )
                {
                    _Stack.Push( node );
                    node = node.Right; //---(_Reverse ? node.Right : node.Left);
                }
                return (true);
            }
            public void Dispose() { }

            public T Current
            {
                [M(O.AggressiveInlining)] get
                {
                    if ( _Current != null )
                    {
                        return (_Current.Item);
                    }
                    return (default);
                }
            }
            object IEnumerator.Current
            {
                get
                {
                    if ( _Current == null ) throw (new InvalidOperationException( "SR.InvalidOperation_EnumOpCantHappen" ));
                    return (_Current.Item);
                }
            }

            internal void Reset()
            {
                _Stack.Clear();
                Intialize_Reverse();
            }
            void IEnumerator.Reset() => Reset();
        }
        #endregion

        #region [.Tree Specific Operations.]
        private static Node GetSibling( Node node, Node parent )
        {
            if ( parent.Left == node )
            {
                return (parent.Right);
            }
            return (parent.Left);
        }

        // After calling InsertionBalance, we need to make sure current and parent up-to-date.
        // It doesn't matter if we keep grandParent and greatGrantParent up-to-date 
        // because we won't need to split again in the next node.
        // By the time we need to split again, everything will be correctly set.
        private void InsertionBalance( Node current, ref Node parent, Node grandParent, Node greatGrandParent )
        {
#if DEBUG
            Debug.Assert( grandParent != null, "Grand parent cannot be null here!" ); 
#endif
            var parentIsOnRight  = (grandParent.Right == parent);
            var currentIsOnRight = (parent.Right == current);

            Node newChildOfGreatGrandParent;
            if ( parentIsOnRight == currentIsOnRight )
            { // same orientation, single rotation
                newChildOfGreatGrandParent = currentIsOnRight ? RotateLeft( grandParent ) : RotateRight( grandParent );
            }
            else
            {  // different orientation, double rotation
                newChildOfGreatGrandParent = currentIsOnRight ? RotateLeftRight( grandParent ) : RotateRightLeft( grandParent );
                // current node now becomes the child of greatgrandparent 
                parent = greatGrandParent;
            }
            // grand parent will become a child of either parent of current.
            grandParent.IsRed = true;
            newChildOfGreatGrandParent.IsRed = false;

            ReplaceChildOfNodeOrRoot( greatGrandParent, grandParent, newChildOfGreatGrandParent );
        }

        [M(O.AggressiveInlining)] private static bool Is2Node( Node node )
        {
#if DEBUG
            Debug.Assert( node != null, "node cannot be null!" ); 
#endif
            return (IsBlack( node ) && IsNullOrBlack( node.Left ) && IsNullOrBlack( node.Right ));
        }
        [M(O.AggressiveInlining)] private static bool Is4Node( Node node ) => (IsRed( node.Left ) && IsRed( node.Right ));
        [M(O.AggressiveInlining)] private static bool IsBlack( Node node ) => (node != null && !node.IsRed);
        [M(O.AggressiveInlining)] private static bool IsNullOrBlack( Node node ) => (node == null || !node.IsRed);
        [M(O.AggressiveInlining)] private static bool IsRed( Node node ) => (node != null && node.IsRed);
        [M(O.AggressiveInlining)] private static void Merge2Nodes( Node parent, Node child1, Node child2 )
        {
#if DEBUG
            Debug.Assert( IsRed( parent ), "parent must be red" ); 
#endif
            // combing two 2-nodes into a 4-node
            parent.IsRed = false;
            child1.IsRed = true;
            child2.IsRed = true;
        }

        // Replace the child of a parent node. 
        // If the parent node is null, replace the root.        
        [M(O.AggressiveInlining)] private void ReplaceChildOfNodeOrRoot( Node parent, Node child, Node newChild )
        {
            if ( parent != null )
            {
                if ( parent.Left == child )
                {
                    parent.Left = newChild;
                }
                else
                {
                    parent.Right = newChild;
                }
            }
            else
            {
                _Root = newChild;
            }
        }
        // Replace the matching node with its successor.
        private void ReplaceNode( Node match, Node parentOfMatch, Node successor, Node parentOfsuccessor )
        {
            if ( successor == match )
            {  // this node has no successor, should only happen if right child of matching node is null.
#if DEBUG
                Debug.Assert( match.Right == null, "Right child must be null!" ); 
#endif
                successor = match.Left;
            }
            else
            {
#if DEBUG
                Debug.Assert( parentOfsuccessor != null, "parent of successor cannot be null!" );
                Debug.Assert( successor.Left == null, "Left child of successor must be null!" );
                Debug.Assert( (successor.Right == null && successor.IsRed) || ((successor.Right?.IsRed).GetValueOrDefault() && !successor.IsRed), "Successor must be in valid state" ); 
#endif
                if ( successor.Right != null )
                {
                    successor.Right.IsRed = false;
                }

                if ( parentOfsuccessor != match )
                {   // detach successor from its parent and set its right child
                    parentOfsuccessor.Left = successor.Right;
                    successor.Right = match.Right;
                }

                successor.Left = match.Left;
            }

            if ( successor != null )
            {
                successor.IsRed = match.IsRed;
            }

            ReplaceChildOfNodeOrRoot( parentOfMatch, match, successor );
        }

        [M(O.AggressiveInlining)] private Node FindNode( in T item )
        {
            var current = _Root;
            while ( current != null )
            {
                var order = Comparer.Compare( in item, in current.Item );
                if ( order == 0 )
                {
                    return (current);
                }
                else
                {
                    current = (order < 0) ? current.Left : current.Right;
                }
            }
            return (null);
        }

        //used for bithelpers. Note that this implementation is completely different 
        //from the Subset's. The two should not be mixed. This indexes as if the tree were an array.
        //http://en.wikipedia.org/wiki/Binary_Tree#Methods_for_storing_binary_trees
        [M(O.AggressiveInlining)] private int InternalIndexOf( in T item )
        {
            var count = 0;
            var current = _Root;
            while ( current != null )
            {
                var order = Comparer.Compare( in item, in current.Item );
                if ( order == 0 )
                {
                    return (count);
                }
                else
                {
                    count = (2 * count + 1);
                    if ( order < 0 )
                    {
                        current = current.Left;                        
                    }
                    else
                    {
                        current = current.Right;
                        count++;
                    }
                }
            }
            return (-1);
        }

        #region not-used. comm.
        /*private Node FindRange( in T from, in T to ) => FindRange( in from, in to, true, true );
        private Node FindRange( in T from, in T to, bool lowerBoundActive, bool upperBoundActive )
        {
            var current = _Root;
            while ( current != null )
            {
                if ( lowerBoundActive && (0 < Comparer.Compare( in from, in current.Item )) )
                {
                    current = current.Right;
                }
                else if ( upperBoundActive && (Comparer.Compare( in to, in current.Item ) < 0) )
                {
                    current = current.Left;
                }
                else
                {
                    return (current);
                }
            }

            return (null);
        }*/
        #endregion

        [M(O.AggressiveInlining)] private static Node RotateLeft( Node node )
        {
            var x = node.Right;
            node.Right = x.Left;
            x.Left = node;
            return (x);
        }
        [M(O.AggressiveInlining)] private static Node RotateLeftRight( Node node )
        {
            var child      = node.Left;
            var grandChild = child.Right;

            node.Left = grandChild.Right;
            grandChild.Right = node;
            child.Right = grandChild.Left;
            grandChild.Left = child;
            return (grandChild);
        }
        [M(O.AggressiveInlining)] private static Node RotateRight( Node node )
        {
            var x = node.Left;
            node.Left = x.Right;
            x.Right = node;
            return (x);
        }
        [M(O.AggressiveInlining)] private static Node RotateRightLeft( Node node )
        {
            var child      = node.Right;
            var grandChild = child.Left;

            node.Right = grandChild.Left;
            grandChild.Left = node;
            child.Left = grandChild.Right;
            grandChild.Right = child;
            return (grandChild);
        }

        /// <summary>
        /// Testing counter that can track rotations
        /// </summary>
        [M(O.AggressiveInlining)] private static TreeRotation RotationNeeded( Node parent, Node current, Node sibling )
        {
#if DEBUG
            Debug.Assert( IsRed( sibling.Left ) || IsRed( sibling.Right ), "sibling must have at least one red child" ); 
#endif
            if ( IsRed( sibling.Left ) )
            {
                if ( parent.Left == current )
                {
                    return (TreeRotation.RightLeftRotation);
                }
                return (TreeRotation.RightRotation);
            }
            else
            {
                if ( parent.Left == current )
                {
                    return (TreeRotation.LeftRotation);
                }
                return (TreeRotation.LeftRightRotation);
            }
        }

        #region not-used. comm.
        /*
        /// <summary>
        /// Decides whether these sets are the same, given the comparer. If the EC's are the same, we can
        /// just use SetEquals, but if they aren't then we have to manually check with the given comparer
        /// </summary>        
        internal static bool SortedSetEquals( SortedSetByRef< T > set1, SortedSetByRef< T > set2, IComparerByRef< T > comparer )
        {
            // handle null cases first
            if ( set1 == null )
            {
                return (set2 == null);
            }
            else if ( set2 == null )
            {
                // set1 != null
                return (false);
            }

            if ( AreComparersEqual( set1, set2 ) )
            {
                if ( set1.Count != set2.Count )
                    return (false);

                return (set1.SetEquals( set2 ));
            }
            else
            {
                foreach ( var item1 in set1 )
                {
                    var found = false;
                    foreach ( var item2 in set2 )
                    {
                        if ( comparer.Compare( in item1, in item2 ) == 0 )
                        {
                            found = true;
                            break;
                        }
                    }
                    if ( !found )
                        return (false);
                }
                return (true);
            }
        }
        */
        #endregion

        //This is a little frustrating because we can't support more sorted structures
        private static bool AreComparersEqual( SortedSetByRef< T > set1, SortedSetByRef< T > set2 ) => set1.Comparer.Equals( set2.Comparer );

        [M(O.AggressiveInlining)] private static void Split4Node( Node node )
        {
            node.IsRed       = true;
            node.Left.IsRed  = false;
            node.Right.IsRed = false;
        }
        #endregion

        #region [.ISet Members.]
        bool ISet< T >.Add( T item ) => AddIfNotPresent( in item );

        /// <summary>
        /// Transform this set into its union with the IEnumerable OTHER            
        ///Attempts to insert each element and rejects it if it exists.          
        /// NOTE: The caller object is important as UnionWith uses the Comparator 
        ///associated with THIS to check equality                                
        /// Throws ArgumentNullException if OTHER is null                         
        /// </summary>
        public void UnionWith( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( other is SortedSetByRef< T > ss )
            {
                if ( Count == 0 )
                {
                    var dummy = new SortedSetByRef< T >( Comparer, ss );
                    _Root = dummy._Root;
                    Count = dummy.Count;
                    /*_version++;*/
                    return;
                }

                if ( AreComparersEqual( this, ss ) && ((this.Count >> 1) < ss.Count) )
                { //this actually hurts if N is much greater than M the /2 is arbitrary
                    //first do a merge sort to an array.
                    var merged = new T[ ss.Count + this.Count ];
                    var c = 0;
                    var mine        = this.GetEnumerator();
                    var theirs      = ss.GetEnumerator();
                    var mineEnded   = !mine.MoveNext();
                    var theirsEnded = !theirs.MoveNext();
                    while ( !mineEnded && !theirsEnded )
                    {
                        var comp = Comparer.Compare( mine.Current, theirs.Current );
                        if ( comp < 0 )
                        {
                            merged[ c++ ] = mine.Current;
                            mineEnded = !mine.MoveNext();
                        }
                        else if ( comp == 0 )
                        {
                            merged[ c++ ] = theirs.Current;
                            mineEnded = !mine.MoveNext();
                            theirsEnded = !theirs.MoveNext();
                        }
                        else
                        {
                            merged[ c++ ] = theirs.Current;
                            theirsEnded = !theirs.MoveNext();
                        }
                    }

                    if ( !mineEnded || !theirsEnded )
                    {
                        var remaining = (mineEnded ? theirs : mine);
                        do
                        {
                            merged[ c++ ] = remaining.Current;
                        } 
                        while ( remaining.MoveNext() );
                    }

                    //now merged has all c elements

                    //safe to gc the root, we have all the elements
                    _Root = null;

                    _Root  = ConstructRootFromSortedArray( merged, 0, c - 1, null );
                    Count = c;
                    /*_version++;*/
                }
            }

            AddAllElements( other );
        }

        private static Node ConstructRootFromSortedArray( T[] arr, int startIndex, int endIndex, Node redNode )
        {
            //what does this do?
            //you're given a sorted array... say 1 2 3 4 5 6 
            //2 cases:
            //    If there are odd # of elements, pick the middle element (in this case 4), and compute
            //    its left and right branches
            //    If there are even # of elements, pick the left middle element, save the right middle element
            //    and call the function on the rest
            //    1 2 3 4 5 6 -> pick 3, save 4 and call the fn on 1,2 and 5,6
            //    now add 4 as a red node to the lowest element on the right branch
            //             3                       3
            //         1       5       ->     1        5
            //           2       6             2     4   6            
            //    As we're adding to the leftmost of the right branch, nesting will not hurt the red-black properties
            //    Leaf nodes are red if they have no sibling (if there are 2 nodes or if a node trickles
            //    down to the bottom

            //the iterative way to do this ends up wasting more space than it saves in stack frames (at
            //least in what i tried)
            //so we're doing this recursively
            //base cases are described below
            var size = endIndex - startIndex + 1;
            if ( size == 0 )
            {
                return (null);
            }
            Node root;
            if ( size == 1 )
            {
                root = new Node( arr[ startIndex ], false );
                if ( redNode != null )
                {
                    root.Left = redNode;
                }
            }
            else if ( size == 2 )
            {
                root = new Node( arr[ startIndex ], false )
                {
                    Right = new Node( arr[ endIndex ] ),
                };
                if ( redNode != null )
                {
                    root.Left = redNode;
                }
            }
            else if ( size == 3 )
            {
                root = new Node( arr[ startIndex + 1 ], false )
                {
                    Left  = new Node( arr[ startIndex ], false ),
                    Right = new Node( arr[ endIndex   ], false ),
                };
                if ( redNode != null )
                {
                    root.Left.Left = redNode;
                }
            }
            else
            {
                var midpt = ((startIndex + endIndex) >> 1);
                root = new Node( arr[ midpt ], false )
                {
                    Left = ConstructRootFromSortedArray( arr, startIndex, midpt - 1, redNode ),
                };
                if ( size % 2 == 0 )
                {
                    root.Right = ConstructRootFromSortedArray( arr, midpt + 2, endIndex, new Node( arr[ midpt + 1 ], true ) );
                }
                else
                {
                    root.Right = ConstructRootFromSortedArray( arr, midpt + 1, endIndex, null );
                }
            }
            return (root);
        }

        /// <summary>
        /// Transform this set into its intersection with the IEnumerable OTHER     
        /// NOTE: The caller object is important as IntersectionWith uses the        
        /// comparator associated with THIS to check equality                        
        /// Throws ArgumentNullException if OTHER is null                         
        /// </summary>
        public void IntersectWith( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
                return;

            //Technically, this would work as well with an ISorted< T > only let this happen if i am also a SortedSet, not a SubSet
            if ( (other is SortedSetByRef< T > ss) && AreComparersEqual( this, ss ) )
            {
                //first do a merge sort to an array.
                var merged = new T[ this.Count ];
                var c      = 0;
                var mine   = this.GetEnumerator();
                var theirs = ss.GetEnumerator();
                bool mineEnded = !mine.MoveNext(), theirsEnded = !theirs.MoveNext();
                T max = Max;

                while ( !mineEnded && !theirsEnded && (Comparer.Compare( theirs.Current, in max ) <= 0) )
                {
                    var comp = Comparer.Compare( mine.Current, theirs.Current );
                    if ( comp < 0 )
                    {
                        mineEnded = !mine.MoveNext();
                    }
                    else if ( comp == 0 )
                    {
                        merged[ c++ ] = theirs.Current;
                        mineEnded = !mine.MoveNext();
                        theirsEnded = !theirs.MoveNext();
                    }
                    else
                    {
                        theirsEnded = !theirs.MoveNext();
                    }
                }

                //now merged has all c elements safe to gc the root, we  have all the elements
                _Root = null;

                _Root = ConstructRootFromSortedArray( merged, 0, c - 1, null );
                Count = c;
            }
            else
            {
                //TODO: Perhaps a more space-conservative way to do this
                var toSave = new List< T >( Count );
                foreach ( var item in other )
                {
                    if ( Contains( in item ) )
                    {
                        toSave.Add( item );
                    }
                }

                if ( toSave.Count < Count )
                {
                    Clear();
                    AddAllElements( toSave );
                }
            }
        }

        /// <summary>
        ///  Transform this set into its complement with the IEnumerable OTHER       
        ///  NOTE: The caller object is important as ExceptWith uses the        
        ///  comparator associated with THIS to check equality                        
        ///  Throws ArgumentNullException if OTHER is null                           
        /// </summary>
        public void ExceptWith( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
                return;

            if ( other == this )
            {
                Clear();
                return;
            }

            if ( (other is SortedSetByRef< T > ss) && AreComparersEqual( this, ss ) )
            {
                //outside range, no point doing anything               
                if ( !(Comparer.Compare( ss.Max, Min ) < 0 || Comparer.Compare( ss.Min, Max ) > 0) )
                {
                    T min = Min;
                    T max = Max;
                    foreach ( var item in other )
                    {
                        if ( Comparer.Compare( in item, in min ) < 0 )
                            continue;
                        if ( Comparer.Compare( in item, in max ) > 0 )
                            break;
                        Remove( in item );
                    }
                }
            }
            else
            {
                RemoveAllElements( other );
            }
        }

        /// <summary>
        ///  Transform this set so it contains elements in THIS or OTHER but not both
        ///  NOTE: The caller object is important as SymmetricExceptWith uses the        
        ///  comparator associated with THIS to check equality                        
        ///  Throws ArgumentNullException if OTHER is null                           
        /// </summary>
        public void SymmetricExceptWith( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
            {
                UnionWith( other );
                return;
            }

            if ( other == this )
            {
                Clear();
                return;
            }

            if ( (other is SortedSetByRef< T > ss) && AreComparersEqual( this, ss ) )
            {
                SymmetricExceptWithSameEC( ss );
            }
            else
            {
                throw (new NotImplementedException());

                /*
                T[] elements = EnumerableHelpers.ToArray( other, out var length );
                Array.Sort( elements, 0, length, Comparer );
                SymmetricExceptWithSameEC( elements, length );
                //*/
            }
        }
        private void SymmetricExceptWithSameEC( SortedSetByRef< T > other )
        {
#if DEBUG
            Debug.Assert( other != null );
            Debug.Assert( AreComparersEqual( this, other ) ); 
#endif
            foreach ( var item in other )
            {
                //yes, it is classier to say
                //if (!this.Remove(item))this.Add(item);
                //but this ends up saving on rotations                    
                if ( Contains( in item ) )
                {
                    Remove( in item );
                }
                else
                {
                    Add( in item );
                }
            }
        }
        //OTHER must be a sorted array
        private void SymmetricExceptWithSameEC( T[] other, int count )
        {
#if DEBUG
            Debug.Assert( other != null );
            Debug.Assert( 0 <= count && count <= other.Length ); 
#endif
            if ( count == 0 )
            {
                return;
            }

            var last = other[ 0 ];
            for ( int i = 0; i < count; i++ )
            {
                while ( (i < count) && (i != 0) && (Comparer.Compare( in other[ i ], in last ) == 0) )
                {
                    i++;
                }
                if ( count <= i )
                {
                    break;
                }
                ref readonly var x = ref other[ i ];
                if ( Contains( in x ) )
                {
                    Remove( in x );
                }
                else
                {
                    Add( in x );
                }
                last = x;
            }
        }

        /// <summary>
        /// Checks whether this Tree is a subset of the IEnumerable other
        /// </summary>
        public bool IsSubsetOf( IEnumerable< T > other )
        {
            throw (new NotImplementedException());

            /*
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
                return (true);
            */

            /*
            var asSorted = other as SortedSetByRef< T >;
            if ( (asSorted != null) && AreComparersEqual( this, asSorted ) )
            {
                if ( asSorted.Count < Count )
                    return (false);
                return (IsSubsetOfSortedSetWithSameEC( asSorted ));
            }
            //*/

            /*
            //worst case: mark every element in my set and see if I've counted all O(MlogN)
            ElementCount result = CheckUniqueAndUnfoundElements( other, false );
            return ((result.UniqueCount == Count) && (0 <= result.UnfoundCount));
            */
        }

        /// <summary>
        /// Checks whether this Tree is a proper subset of the IEnumerable other
        /// </summary>
        public bool IsProperSubsetOf( IEnumerable< T > other )
        {
            throw (new NotImplementedException());

            /*
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( other is ICollection coll )
            {
                if ( Count == 0 )
                    return (coll.Count > 0);
            }
            */

            /*
            //another for sorted sets with the same comparer
            var asSorted = other as SortedSetByRef< T >;
            if ( (asSorted != null) && AreComparersEqual( this, asSorted ) )
            {
                if ( asSorted.Count <= Count)
                    return (false);
                return (IsSubsetOfSortedSetWithSameEC( asSorted ));
            }
            //*/

            /*
            //worst case: mark every element in my set and see if I've counted all O(MlogN).
            ElementCount result = CheckUniqueAndUnfoundElements( other, false );
            return ((result.UniqueCount == Count) && (0 < result.UnfoundCount));
            */
        }

        /// <summary>
        /// Checks whether this Tree is a super set of the IEnumerable other
        /// </summary>
        public bool IsSupersetOf( IEnumerable< T > other )
        {
            throw (new NotImplementedException());

            /*
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( (other is ICollection coll) && (coll.Count == 0) )
            {
                return (true);
            }
            */

            //do it one way for HashSets
            //another for sorted sets with the same comparer

            /*
            var asSorted = other as SortedSetByRef< T >;
            if ( (asSorted != null) && AreComparersEqual( this, asSorted ) )
            {
                if ( Count < asSorted.Count )
                {
                    return (false);
                }
                var pruned = GetViewBetween( asSorted.Min, asSorted.Max );
                foreach ( var item in asSorted )
                {
                    if ( !pruned.Contains( item ) )
                    {
                        return (false);
                    }
                }
                return (true);
            }
            //*/

            //and a third for everything else
            //---return (ContainsAllElements( other ));
        }

        /// <summary>
        /// Checks whether this Tree is a proper super set of the IEnumerable other
        /// </summary>
        public bool IsProperSupersetOf( IEnumerable< T > other )
        {
            throw (new NotImplementedException());

            /*
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
                return (false);

            if ( (other is ICollection coll) && (coll.Count == 0) )
            {
                return (true);
            }
            */


            /*
            //another way for sorted sets
            var asSorted = other as SortedSetByRef< T >;
            if ( (asSorted != null) && AreComparersEqual( asSorted, this ) )
            {
                if ( asSorted.Count >= Count )
                {
                    return (false);
                }
                var pruned = GetViewBetween( asSorted.Min, asSorted.Max );
                foreach ( var item in asSorted )
                {
                    if ( !pruned.Contains( item ) )
                    {
                        return (false);
                    }
                }
                return (true);
            }
            //*/

            //worst case: mark every element in my set and see if I've counted all O(MlogN)
            //slight optimization, put it into a HashSet and then check can do it in O(N+M)
            //but slower in better cases + wastes space
            //---ElementCount result = CheckUniqueAndUnfoundElements( other, true );
            //---return (result.UniqueCount < Count && result.UnfoundCount == 0);
        }

        /// <summary>
        /// Checks whether this Tree has all elements in common with IEnumerable other
        /// </summary>
        public bool SetEquals( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( (other is SortedSetByRef< T > ss) && AreComparersEqual( this, ss ) )
            {
                var mine   = GetEnumerator();
                var theirs = ss.GetEnumerator();
                var mineEnded    = !mine.MoveNext();
                var theirsEnded  = !theirs.MoveNext();
                while ( !mineEnded && !theirsEnded )
                {
                    if ( Comparer.Compare( mine.Current, theirs.Current ) != 0 )
                    {
                        return (false);
                    }
                    mineEnded   = !mine.MoveNext();
                    theirsEnded = !theirs.MoveNext();
                }
                return (mineEnded && theirsEnded);
            }

            //worst case: mark every element in my set and see if I've counted all O(N) by size of other            
            ElementCount result = CheckUniqueAndUnfoundElements( other, true );
            return ((result.UniqueCount == Count) && (result.UnfoundCount == 0));
        }

        /// <summary>
        /// Checks whether this Tree has any elements in common with IEnumerable other
        /// </summary>
        public bool Overlaps( IEnumerable< T > other )
        {
            if ( other == null ) throw (new ArgumentNullException( nameof(other) ));

            if ( Count == 0 )
                return (false);

            if ( (other is ICollection< T > coll) && (coll.Count == 0) )
            {
                return (false);
            }

            if ( (other is SortedSetByRef< T > asSorted) && AreComparersEqual( this, asSorted ) && (Comparer.Compare( Min, asSorted.Max ) > 0 || Comparer.Compare( Max, asSorted.Min ) < 0) )
            {
                return (false);
            }

            foreach ( var item in other )
            {
                if ( Contains( in item ) )
                {
                    return (true);
                }
            }
            return (false);
        }

        /// <summary>
        /// This works similar to HashSet's CheckUniqueAndUnfound (description below), except that the bit
        /// array maps differently than in the HashSet. We can only use this for the bulk boolean checks.
        /// 
        /// Determines counts that can be used to determine equality, subset, and superset. This
        /// is only used when other is an IEnumerable and not a HashSet. If other is a HashSet
        /// these properties can be checked faster without use of marking because we can assume 
        /// other has no duplicates.
        /// 
        /// The following count checks are performed by callers:
        /// 1. Equals: checks if unfoundCount = 0 and uniqueFoundCount = Count; i.e. everything 
        /// in other is in this and everything in this is in other
        /// 2. Subset: checks if unfoundCount >= 0 and uniqueFoundCount = Count; i.e. other may
        /// have elements not in this and everything in this is in other
        /// 3. Proper subset: checks if unfoundCount > 0 and uniqueFoundCount = Count; i.e
        /// other must have at least one element not in this and everything in this is in other
        /// 4. Proper superset: checks if unfound count = 0 and uniqueFoundCount strictly less
        /// than Count; i.e. everything in other was in this and this had at least one element
        /// not contained in other.
        /// 
        /// An earlier implementation used delegates to perform these checks rather than returning
        /// an ElementCount struct; however this was changed due to the perf overhead of delegates.
        /// </summary>
        /// <param name="returnIfUnfound">Allows us to finish faster for equals and proper superset
        /// because unfoundCount must be 0.</param>
        /// <returns></returns>
        // <SecurityKernel Critical="True" Ring="0">
        // <UsesUnsafeCode Name="Local bitArrayPtr of type: Int32*" />
        // <ReferencesCritical Name="Method: BitHelper..ctor(System.Int32*,System.Int32)" Ring="1" />
        // <ReferencesCritical Name="Method: BitHelper.IsMarked(System.Int32):System.Boolean" Ring="1" />
        // <ReferencesCritical Name="Method: BitHelper.MarkBit(System.Int32):System.Void" Ring="1" />
        // </SecurityKernel>
        private unsafe ElementCount CheckUniqueAndUnfoundElements( IEnumerable< T > other, bool returnIfUnfound )
        {
            ElementCount result;

            // need special case in case this has no elements. 
            if ( Count == 0 )
            {
                var numElementsInOther = 0;
                foreach ( T item in other )
                {
                    numElementsInOther++;
                    // break right away, all we want to know is whether other has 0 or 1 elements
                    break;
                }
                result.UniqueCount  = 0;
                result.UnfoundCount = numElementsInOther;
                return (result);
            }

            int intArrayLength = BitHelper.ToIntArrayLength( this.Count );
            BitHelper bitHelper;
            if ( intArrayLength <= STACK_ALLOC_THRESHOLD )
            {
                int* bitArrayPtr = stackalloc int[ intArrayLength ];
                bitHelper = new BitHelper( bitArrayPtr, intArrayLength );
            }
            else
            {
                int[] bitArray = new int[ intArrayLength ];
                bitHelper = new BitHelper( bitArray, intArrayLength );
            }

            // count of items in other not found in this
            int unfoundCount = 0;
            // count of unique items in other found in this
            int uniqueFoundCount = 0;

            foreach ( var item in other )
            {
                var index = InternalIndexOf( item );
                if ( 0 <= index )
                {
                    if ( !bitHelper.IsMarked( index ) )
                    {
                        // item hasn't been seen yet
                        bitHelper.MarkBit( index );
                        uniqueFoundCount++;
                    }
                }
                else
                {
                    unfoundCount++;
                    if ( returnIfUnfound )
                    {
                        break;
                    }
                }
            }

            result.UniqueCount  = uniqueFoundCount;
            result.UnfoundCount = unfoundCount;
            return (result);
        }
        public int RemoveWhere( Predicate< T > match )
        {
            if ( match == null ) throw (new ArgumentNullException( nameof(match) ));

            var matches = new List< T >( this.Count );
            BreadthFirstTreeWalk( (n) =>
            {
                if ( match( n.Item ) )
                {
                    matches.Add( n.Item );
                }
                return (true);
            });
            // reverse breadth first to (try to) incur low cost
            var actuallyRemoved = 0;
            for ( var i = matches.Count - 1; i >= 0; i-- )
            {
                if ( Remove( matches[ i ] ) )
                {
                    actuallyRemoved++;
                }
            }

            return (actuallyRemoved);
        }
        #endregion

        #region [.ISorted Members.]
        public T Min
        {
            get
            {
                if ( _Root == null )
                {
                    return (default);
                }

                var current = _Root;
                while ( current.Left != null )
                {
                    current = current.Left;
                }

                return (current.Item);
            }
        }
        public T Max
        {
            get
            {
                if ( _Root == null )
                {
                    return (default);
                }

                var current = _Root;
                while ( current.Right != null )
                {
                    current = current.Right;
                }

                return (current.Item);
            }
        }

        public IEnumerable< T > Reverse()
        {            
            for ( var e = new Enumerator_4_Reverse( this ); e.MoveNext(); )
            {
                yield return (e.Current);
            }
        }
        #endregion

        #region [.Helper Classes.]
        /// <summary>
        /// 
        /// </summary>
        private delegate bool TreeWalkPredicate< X >( Node node );

        /// <summary>
        /// 
        /// </summary>
        private enum TreeRotation
        {
            LeftRotation      = 1,
            RightRotation     = 2,
            RightLeftRotation = 3,
            LeftRightRotation = 4,
        }

        /// <summary>
        /// 
        /// </summary>
        public sealed class Node
        {
            public Node Left;
            public Node Right;            
            public T    Item;
            public bool IsRed;

            [M(O.AggressiveInlining)] public Node( in T item )
            {
                // The default color will be red, we never need to create a black node directly.                
                Item  = item;
                IsRed = true;
            }
            [M(O.AggressiveInlining)] public Node( in T item, bool isRed )
            {
                // The default color will be red, we never need to create a black node directly.                
                Item  = item;
                IsRed = isRed;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private struct ElementCount
        {
            internal int UniqueCount;
            internal int UnfoundCount;
        }
        #endregion

        #region [.misc.]
        // used for set checking operations (using enumerables) that rely on counting
        [M(O.AggressiveInlining)] private static int log2( int value )
        {
            var c = 0;
            while ( 0 < value )
            {
                c++;
                value >>= 1;
            }
            return (c);
        }
        #endregion
    }

    #region not-used. comm.
    /*
    /// <summary>
    /// A class that generates an IEqualityComparer for this SortedSet. Requires that the definition of
    /// equality defined by the IComparer for this SortedSet be consistent with the default IEqualityComparer
    /// for the type T. If not, such an IEqualityComparer should be provided through the constructor.
    /// </summary>    
    internal sealed class SortedSetByRefEqualityComparer< T > : IEqualityComparer< SortedSetByRef< T > > where T : struct
    {
        private readonly IComparerByRef< T > _Comparer;
        private readonly IEqualityComparerByRef< T > _MemberEqualityComparer;

        /// <summary>
        /// Create a new SetEqualityComparer, given a comparer for member order and another for member equality (these must be consistent in their definition of equality)
        /// </summary>        
        public SortedSetByRefEqualityComparer( IComparerByRef< T > comparer, IEqualityComparerByRef< T > memberEqualityComparer )
        {
            _Comparer               = comparer               ?? throw (new ArgumentNullException( nameof(comparer) ));
            _MemberEqualityComparer = memberEqualityComparer ?? throw (new ArgumentNullException( nameof(memberEqualityComparer) ));
        }

        // using comparer to keep equals properties in tact; don't want to choose one of the comparers
        public bool Equals( SortedSetByRef< T > x, SortedSetByRef< T > y ) => SortedSetByRef< T >.SortedSetEquals( x, y, _Comparer );

        //IMPORTANT: this part uses the fact that GetHashCode() is consistent with the notion of equality in the set
        public int GetHashCode( SortedSetByRef< T > obj )
        {
            var hashCode = 0;
            if ( obj != null )
            {
                foreach ( var t in obj )
                {
                    hashCode ^= (_MemberEqualityComparer.GetHashCode( in t ) & 0x7FFFFFFF);
                }
            } // else returns hashcode of 0 for null HashSets
            return (hashCode);
        }

        // Equals method for the comparer itself. 
        public override bool Equals( object obj ) => (obj is SortedSetByRefEqualityComparer< T > comparer) && (_Comparer == comparer._Comparer);
        public override int GetHashCode() => (_Comparer.GetHashCode() ^ _MemberEqualityComparer.GetHashCode());
    }
    //*/
    #endregion
}