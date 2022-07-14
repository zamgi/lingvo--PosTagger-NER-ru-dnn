using System;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public abstract class OpConstraint
    {
        public abstract bool SatisfiedFor( object[] args );
    }
    /// <summary>
    /// 
    /// </summary>
    public class ArgCountConstraint : OpConstraint
    {
        private readonly int _ArgCount;
        public ArgCountConstraint( int argCount ) => _ArgCount = argCount;
        public override bool SatisfiedFor( object[] args ) => args.Length == _ArgCount;
    }
    /// <summary>
    /// 
    /// </summary>
    public class ArgTypeConstraint : OpConstraint
    {
        private readonly int  _ArgIndex;
        private readonly Type _RequiredType;
        public ArgTypeConstraint( int argIndex, Type requiredType )
        {
            _ArgIndex     = argIndex;
            _RequiredType = requiredType;
        }
        public override bool SatisfiedFor( object[] args ) => _RequiredType.IsAssignableFrom( args[ _ArgIndex ].GetType() );
    }
    /// <summary>
    /// 
    /// </summary>
    public class ArgStorageTypeConstraint : OpConstraint
    {
        private readonly int  _ArgIndex;
        private readonly Type _RequiredType;
        private readonly bool _AllowNull;

        public ArgStorageTypeConstraint( int argIndex, Type requiredType, bool allowNull = true )
        {
            _ArgIndex     = argIndex;
            _RequiredType = requiredType;
            _AllowNull    = allowNull;
        }

        public override bool SatisfiedFor( object[] args )
        {
            var arg = args[ _ArgIndex ];
            if ( _AllowNull && (arg == null) )
            {
                return (true);
            }
            else if ( !_AllowNull && (arg == null) )
            {
                return (false);
            }

            Storage argStorage = ((Tensor) arg).Storage;
            return (_RequiredType.IsAssignableFrom( argStorage.GetType() ));
        }
    }
}
