using System;
using System.Collections.Generic;

namespace Lingvo.PosTagger.Tensors.Cuda.Util
{
    /// <summary>
    /// 
    /// </summary>
    public class PooledObject< T > : IDisposable
    {
        private readonly Action< PooledObject< T > > _DisposeAction;
        private readonly T _Value;

        private bool _IsDisposed;

        public PooledObject( T value, Action< PooledObject< T > > onDispose )
        {
            _DisposeAction = onDispose ?? throw (new ArgumentNullException( "onDispose" ));
            _Value         = value;
        }

        public T Value
        {
            get
            {
                if ( _IsDisposed ) throw (new ObjectDisposedException( ToString() ));
                return (_Value);
            }
        }

        public void Dispose()
        {
            if ( !_IsDisposed )
            {
                _DisposeAction( this );
                _IsDisposed = true;
            }
            else
            {
                throw (new ObjectDisposedException( ToString() ));
            }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class ObjectPool< T > : IDisposable
    {
        private readonly Func< T > _Constructor;
        private readonly Action< T > _Destructor;
        private readonly Stack< T > _FreeList = new Stack<T>();
        private bool _IsDisposed;

        public ObjectPool( int initialSize, Func<T> constructor, Action<T> destructor )
        {
            _Constructor = constructor ?? throw (new ArgumentNullException( "constructor" ));
            _Destructor  = destructor  ?? throw (new ArgumentNullException( "destructor" ));

            for ( int i = 0; i < initialSize; ++i )
            {
                _FreeList.Push( constructor() );
            }
        }

        public void Dispose()
        {
            if ( !_IsDisposed )
            {
                _IsDisposed = true;
                foreach ( T t in _FreeList )
                {
                    _Destructor( t );
                }
                _FreeList.Clear();
            }
        }

        public PooledObject<T> Get()
        {
            T value = ((0 < _FreeList.Count) ? _FreeList.Pop() : _Constructor());
            return (new PooledObject< T >( value, Release ));
        }

        private void Release( PooledObject< T > handle ) => _FreeList.Push( handle.Value );
    }
}
