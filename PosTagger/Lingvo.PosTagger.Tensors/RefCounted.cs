using System;
using System.Threading;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// Provides a thread safe reference counting implementation. Inheritors need only implement the Destroy() method,
    /// which will be called when the reference count reaches zero. The reference count automatically starts at 1.
    /// </summary>
    public abstract class RefCounted
    {
        private int _RefCount;

        /// <summary>
        /// Construct a new reference counted object. The reference count automatically starts at 1.
        /// </summary>
        public RefCounted() => _RefCount = 1;
        ~RefCounted()
        {
            if ( 0 < _RefCount )
            {
                Destroy();
                _RefCount = 0;
            }
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected abstract void Destroy();

        /// <summary>
        /// Returns true if the object has already been destroyed; false otherwise.
        /// </summary>
        /// <returns>true if the object is destroyed; false otherwise.</returns>
        protected bool IsDestroyed() => (_RefCount == 0);

        /// <summary>
        /// Throws an exception if the object has been destroyed, otherwise does nothing.
        /// </summary>
        protected void ThrowIfDestroyed()
        {
            if ( IsDestroyed() )
            {
                throw (new InvalidOperationException( "Reference counted object has been destroyed." ));
            }
        }
        protected int GetCurrentRefCount() => _RefCount;

        /// <summary>
        /// Increments the reference count. If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void AddRef()
        {
            int curRefCount;
            int original;
            var spin = new SpinWait();
            while ( true )
            {
                curRefCount = _RefCount;
                if ( curRefCount == 0 )
                {
                    throw (new InvalidOperationException( "Cannot AddRef - object has already been destroyed." ));
                }

                int desiredRefCount = curRefCount + 1;
                original = Interlocked.CompareExchange( ref _RefCount, desiredRefCount, curRefCount );
                if ( original == curRefCount )
                {
                    break;
                }

                spin.SpinOnce();
            }
        }

        /// <summary>
        /// Decrements the reference count. If the reference count reaches zero, the object is destroyed.
        /// If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void Release()
        {
            int original;
            int curRefCount;
            var spin = new SpinWait();
            while ( true )
            {
                curRefCount = _RefCount;
                if ( curRefCount == 0 )
                {
                    throw (new InvalidOperationException( "Cannot release object - object has already been destroyed." ));
                }

                int desiredRefCount = _RefCount - 1;
                original = Interlocked.CompareExchange( ref _RefCount, desiredRefCount, curRefCount );
                if ( original == curRefCount )
                {
                    break;
                }

                spin.SpinOnce();
            }

            if ( _RefCount <= 0 )
            {
                Destroy();
            }
        }
    }
}
