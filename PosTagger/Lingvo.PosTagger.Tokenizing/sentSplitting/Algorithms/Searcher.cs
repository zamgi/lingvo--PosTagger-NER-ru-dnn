using System.Collections.Generic;
using System.Linq;
#if DEBUG
using System.Diagnostics;
#endif
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.SentSplitting
{
    /// <summary>
    /// 
    /// </summary>
    internal readonly struct ngram_t< TValue >
    {
        [M(O.AggressiveInlining)] public ngram_t( string[] _words, TValue _value )
        {
            words = _words;
            value = _value;
        }
        public string[] words { [M(O.AggressiveInlining)] get; }
        public TValue   value { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString() => ('\'' + string.Join( "' '", words ) + "' (" + words.Length + "), '" + value + "'"); 
#endif
    }

    /// <summary>
    /// 
    /// </summary>
    internal readonly struct SearchResult< TValue >
    {
        [M(O.AggressiveInlining)] public SearchResult( int startIndex, int length, TValue value )
        {
            StartIndex = startIndex;
            Length     = length;
            v          = value;
        }

        public int    StartIndex { [M(O.AggressiveInlining)] get; }
        public int    Length     { [M(O.AggressiveInlining)] get; }
        public TValue v          { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString()
        {
            var s = v.ToString();
            if ( string.IsNullOrEmpty( s ) )
            {
                return ("[" + StartIndex + ":" + Length + "]");
            }
            return ("[" + StartIndex + ":" + Length + "], value: '" + s + "'");
        }
#endif
    }

    /// <summary>
    /// 
    /// </summary>
    internal readonly struct SearchResultOfHead2Left< TValue >
    {
        [M(O.AggressiveInlining)] public SearchResultOfHead2Left( ss_word_t lastWord, int length, TValue value )
        {
            LastWord = lastWord;
            Length   = length;
            v        = value;
        }
        [M(O.AggressiveInlining)] public SearchResultOfHead2Left( ss_word_t lastWord, in ngram_t< TValue > ngram )
        {
            LastWord = lastWord;
            Length   = ngram.words.Length;
            v        = ngram.value;
        }

        public ss_word_t LastWord { [M(O.AggressiveInlining)] get; }
        public int       Length   { [M(O.AggressiveInlining)] get; }
        public TValue    v        { [M(O.AggressiveInlining)] get; }
#if DEBUG
        public override string ToString()
        {
            var s = v.ToString();
            if ( string.IsNullOrEmpty( s ) )
            {
                return ("[0:" + Length + "]");
            }
            return ("[0:" + Length + "], value: '" + s + "'");
        }
#endif
    }

    /// <summary>
    /// Class for searching string for one or multiple keywords using efficient Aho-Corasick search algorithm
    /// </summary>
    internal sealed class Searcher< TValue >
    {
        /// <summary>
        /// Tree node representing character and its transition and failure function
        /// </summary>
        private sealed class TreeNode
        {
            /// <summary>
            /// 
            /// </summary>
            private sealed class ngram_EqualityComparer : IEqualityComparer< ngram_t< TValue > >
            {
                public static ngram_EqualityComparer Inst { get; } = new ngram_EqualityComparer();
                private ngram_EqualityComparer() { }
                public bool Equals( ngram_t< TValue > x, ngram_t< TValue > y )
                {
                    var len = x.words.Length;
                    if ( len != y.words.Length )
                    {
                        return (false);
                    }

                    for ( var i = 0; i < len; i++ )
                    {
                        if ( !string.Equals( x.words[ i ], y.words[ i ] ) )
                        {
                            return (false);
                        }
                    }
                    return (true);
                }
                public int GetHashCode( ngram_t< TValue > obj ) => obj.words.Length;
            }

            public static TreeNode BuildTree( IEnumerable< ngram_t< TValue > > ngrams )
            {
                var transitions_root_nodes = default(Map< string, TreeNode >.ReadOnlyCollection4Values);
                var transitions_nodes      = default(Map< string, TreeNode >.ReadOnlyCollection4Values);
                var ngs                    = default(IReadOnlyCollection< ngram_t< TValue > >);

                // Build keyword tree and transition function
                var root = new TreeNode( null, null );
                foreach ( var ngram in ngrams )
                {
                    // add pattern to tree
                    var node = root;
                    foreach ( string word in ngram.words )
                    {
                        var nodeNew = node.GetTransition( word );
                        if ( nodeNew == null )
                        {
                            nodeNew = new TreeNode( node, word );
                            node.AddTransition( nodeNew );
                        }
                        node = nodeNew;
                    }
                    node.AddNgram( in ngram );
                }

                // Find failure functions
                var nodes = new List< TreeNode >();
                // level 1 nodes - fail to root node
                if ( root.TryGetAllTransitions( ref transitions_root_nodes ) )
                {
                    nodes.Capacity = transitions_root_nodes.Count;

                    foreach ( var node in transitions_root_nodes )
                    {
                        node.Failure = root;
                        if ( node.TryGetAllTransitions( ref transitions_nodes ) )
                        {
                            foreach ( var trans in transitions_nodes )
                            {
                                nodes.Add( trans );
                            }
                        }
                    }
                }

                // other nodes - using BFS
                while ( nodes.Count != 0 )
                {
                    var newNodes = new List< TreeNode >( nodes.Count );
                    foreach ( var node in nodes )
                    {
                        var r = node.Parent.Failure;
                        var word = node.Word;
                        
                        while ( (r != null) && !r.ContainsTransition( word ) )
                        {
                            r = r.Failure;
                        }
                        if ( r == null )
                        {
                            node.Failure = root;
                        }
                        else
                        {
                            var nf = node.Failure = r.GetTransition( word );
                            if ( (nf != null) && nf.TryGetNgrams( ref ngs ) )
                            {
                                foreach ( var ng in ngs )
                                {
                                    node.AddNgram( in ng );
                                }
                            }
                        }

                        // add child nodes to BFS list 
                        if ( node.TryGetAllTransitions( ref transitions_nodes ) )
                        {
                            foreach ( var child in transitions_nodes )
                            {
                                newNodes.Add( child );
                            }
                        }
                    }
                    nodes = newNodes;
                }
                root.Failure = root;

                return (root);
            }

            #region [.ctor() & methods.]
            /// <summary>
            /// Initialize tree node with specified character
            /// </summary>
            /// <param name="parent">Parent node</param>
            /// <param name="word">word</param>
            public TreeNode( TreeNode parent, string word )
            {
                Word   = word;
                Parent = parent;                
            }

            /// <summary>
            /// Adds pattern ending in this node
            /// </summary>
            /// <param name="ngram">Pattern</param>
            public void AddNgram( in ngram_t< TValue > ngram )
            {
                if ( _Ngrams == null )
                {
                    _Ngrams = new Set< ngram_t< TValue > >( ngram_EqualityComparer.Inst );
                }
                _Ngrams.Add( ngram );
            }

            /// <summary>
            /// Adds trabsition node
            /// </summary>
            /// <param name="node">Node</param>
            public void AddTransition( TreeNode node )
            {
                if ( _TransDict == null )
                {
                    _TransDict = new Map< string, TreeNode >();
                }
                _TransDict.Add( node.Word, node );
            }

            /// <summary>
            /// Returns transition to specified character (if exists)
            /// </summary>
            /// <param name="word">word</param>
            /// <returns>Returns TreeNode or null</returns>
            [M(O.AggressiveInlining)] public TreeNode GetTransition( string word ) => ((_TransDict != null) && _TransDict.TryGetValue( word, out var node )) ? node : null;

            /// <summary>
            /// Returns true if node contains transition to specified character
            /// </summary>
            /// <param name="c">Character</param>
            /// <returns>True if transition exists</returns>
            [M(O.AggressiveInlining)] public bool ContainsTransition( string word ) => ((_TransDict != null) && _TransDict.ContainsKey( word ));
            #endregion

            #region [.props.]
            private Map< string, TreeNode > _TransDict;
            private Set< ngram_t< TValue > > _Ngrams;

            /// <summary>
            /// Character
            /// </summary>
            public string Word { [M(O.AggressiveInlining)] get; private set; }

            /// <summary>
            /// Parent tree node
            /// </summary>
            public TreeNode Parent { [M(O.AggressiveInlining)] get; private set; }

            /// <summary>
            /// Failure function - descendant node
            /// </summary>
            public TreeNode Failure { [M(O.AggressiveInlining)] get; internal set; }

            /// <summary>
            /// Transition function - list of descendant nodes
            /// </summary>
            [M(O.AggressiveInlining)] public bool TryGetAllTransitions( ref Map< string, TreeNode >.ReadOnlyCollection4Values trs )
            {
                if ( _TransDict != null )
                {
                    trs = _TransDict.GetValues();
                    return (true);
                }
                return (false);
            }

            [M(O.AggressiveInlining)] public bool TryGetNgrams( ref IReadOnlyCollection< ngram_t< TValue > > ngs )
            {
                if ( _Ngrams != null )
                {
                    ngs = _Ngrams;
                    return (true);
                }
                return (false);
            }
            /// <summary>
            /// Returns list of patterns ending by this letter
            /// </summary>
            public IReadOnlyCollection< ngram_t< TValue > > Ngrams { [M(O.AggressiveInlining)] get => _Ngrams; }
            public bool HasNgrams { [M(O.AggressiveInlining)] get => (_Ngrams != null); }
            #endregion
#if DEBUG
            public override string ToString() => ( ((Word != null) ? ('\'' + Word + '\'') : "ROOT") + ", transitions(descendants): " + ((_TransDict != null) ? _TransDict.Count : 0) + ", ngrams: " + ((_Ngrams != null) ? _Ngrams.Count : 0));
#endif
        }

        /// <summary>
        /// 
        /// </summary>
        private sealed class SearchResultComparer : IComparerByRef< SearchResult< TValue > >
        {
            public static SearchResultComparer Inst { get; } = new SearchResultComparer();
            private SearchResultComparer() { }
            public int Compare( in SearchResult< TValue > x, in SearchResult< TValue > y )
            {
                var d = y.Length - x.Length;
                if ( d != 0 )
                    return (d);

                return (x.StartIndex - y.StartIndex);

                #region comm.
                //var d = x.StartIndex - y.StartIndex;
                //if ( d != 0 )
                //return (d);

                //return (y.Value - x.Value); 
                #endregion
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private sealed class SearchResultOfHead2LeftIComparer : IComparerByRef< SearchResultOfHead2Left< TValue > >
        {
            public static SearchResultOfHead2LeftIComparer Inst { get; } = new SearchResultOfHead2LeftIComparer();
            private SearchResultOfHead2LeftIComparer() { }
            public int Compare( in SearchResultOfHead2Left< TValue > x, in SearchResultOfHead2Left< TValue > y ) => (y.Length - x.Length);
        }

        /// <summary>
        /// 
        /// </summary>
        private struct Finder
        {
            private TreeNode _Root;
            private TreeNode _Node;
            [M(O.AggressiveInlining)] public static Finder Create( TreeNode root ) => new Finder() { _Root = root, _Node = root };

            [M(O.AggressiveInlining)] public TreeNode Find( string word )
            {
                TreeNode transNode;
                do
                {
                    transNode = _Node.GetTransition( word );
                    if ( _Node == _Root )
                    {
                        break;
                    }
                    if ( transNode == null )
                    {
                        _Node = _Node.Failure;
                    }
                }
                while ( transNode == null );
                if ( transNode != null )
                {
                    _Node = transNode;
                }
                return (_Node);
            }
        }

        #region [.private field's.]
        private static readonly SearchResult< TValue >[]            EMPTY_RESULT_1 = new SearchResult< TValue >[ 0 ];
        private static readonly SearchResultOfHead2Left< TValue >[] EMPTY_RESULT_2 = new SearchResultOfHead2Left< TValue >[ 0 ];
        /// <summary>
        /// Root of keyword tree
        /// </summary>
        private TreeNode _Root;
        #endregion

        #region [.ctor().]
        /// <summary>
        /// Initialize search algorithm (Build keyword tree)
        /// </summary>
        /// <param name="keywords">Keywords to search for</param>
        internal Searcher( IList< ngram_t< TValue > > ngrams )
        {
            _Root = TreeNode.BuildTree( ngrams );
            NgramMaxLength = (0 < ngrams.Count) ? ngrams.Max( ngram => ngram.words.Length ) : 0;
        }
        #endregion

        #region [.public method's & properties.]
        internal int NgramMaxLength { [M(O.AggressiveInlining)] get; }

        internal IReadOnlyCollection< SearchResult< TValue > > FindAll( DirectAccessList< ss_word_t > words )
        {
            var ss = default(SortedSetByRef< SearchResult< TValue > >);
            var finder = Finder.Create( _Root );

            for ( int index = 0, len = words.Count; index < len; index++ )
            {
                var node = finder.Find( words._Items[ index ].valueOriginal );
                if ( node.HasNgrams )
                {
                    if ( ss == null ) ss = new SortedSetByRef< SearchResult< TValue > >( SearchResultComparer.Inst );
                    
                    foreach ( var ngram in node.Ngrams )
                    {
                        var r = ss.Add( new SearchResult< TValue >( index - ngram.words.Length + 1, ngram.words.Length, ngram.value ) );
#if DEBUG
                        Debug.Assert( r ); 
#endif
                    }
                }
            }
            if ( ss != null )
            {
                return (ss);
            }
            return (EMPTY_RESULT_1);
        }
        internal IReadOnlyCollection< SearchResultOfHead2Left< TValue > > FindOfHead2Left( ss_word_t headWord )
        {
            var ss = default(SortedSetByRef< SearchResultOfHead2Left< TValue > >);
            var finder = Finder.Create( _Root );            
            int index = 0;

            for ( var word = headWord; word != null; word = word.next )
            {
                var node = finder.Find( word.valueOriginal );
                if ( node.HasNgrams )
                {
                    foreach ( var ngram in node.Ngrams )
                    {
                        var wordIndex = index - ngram.words.Length + 1;
                        if ( wordIndex == 0 )
                        {
                            if ( ss == null ) ss = new SortedSetByRef< SearchResultOfHead2Left< TValue > >( SearchResultOfHead2LeftIComparer.Inst );

                            var r = ss.Add( new SearchResultOfHead2Left< TValue >( word, in ngram ) );
#if DEBUG
                            Debug.Assert( r ); 
#endif
                        }
                    }
                }
                index++;
            }
            if ( ss != null )
            {
                return (ss);
            }
            return (EMPTY_RESULT_2);
        }
        internal bool TryFindOfHead2LeftFirst( ss_word_t headWord, out SearchResultOfHead2Left< TValue > result )
        {
            result = default;
            var finder = Finder.Create( _Root );            
            int index = 0;

            for ( var word = headWord; word != null; word = word.next )
            {
                var node = finder.Find( word.valueOriginal );
                if ( node.HasNgrams )
                {
                    foreach ( var ngram in node.Ngrams )
                    {
                        var wordIndex = index - ngram.words.Length + 1;
                        if ( wordIndex == 0 )
                        {
                            var sr = new SearchResultOfHead2Left< TValue >( word, in ngram );
                            if ( result.Length < sr.Length ) result = sr;
                        }
                    }
                }
                index++;
            }
            return (result.Length != 0);
        }
        #endregion
#if DEBUG
        public override string ToString() => $"[{_Root}]";
#endif
    }
}
