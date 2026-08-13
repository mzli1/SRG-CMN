 override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<MapArray>* out) { return FinishTyped(out); }

  /// \brief Vector append
  ///
  /// If passed, valid_bytes is of equal length to values, and any zero byte
  /// will be considered as a null for that slot
  Status AppendValues(const int32_t* offsets, int64_t length,
                      const uint8_t* valid_bytes = NULLPTR);

  /// \brief Start a new variable-length map slot
  ///
  /// This function should be called before beginning to append elements to the
  /// key and item builders
  Status Append();

  Status AppendNull() final;

  Status AppendNulls(int64_t length) final;

  Status AppendEmptyValue() final;

  Status AppendEmptyValues(int64_t length) final;

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override {
    const auto* offsets = array.GetValues<int32_t>(1);
    static_assert(internal::may_have_validity_bitmap(MapType::type_id));
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    for (int64_t row = offset; row < offset + length; row++) {
      const bool is_valid = !validity || bit_util::GetBit(validity, array.offset + row);
      if (is_valid) {
        ARROW_RETURN_NOT_OK(Append());
        const int64_t slot_length = offsets[row + 1] - offsets[row];
        // Add together the inner StructArray offset to the Map/List offset
        int64_t key_value_offset = array.child_data[0].offset + offsets[row];
        ARROW_RETURN_NOT_OK(key_builder_->AppendArraySlice(
            array.child_data[0].child_data[0], key_value_offset, slot_length));
        ARROW_RETURN_NOT_OK(item_builder_->AppendArraySlice(
            array.child_data[0].child_data[1], key_value_offset, slot_length));
      } else {
        ARROW_RETURN_NOT_OK(AppendNull());
      }
    }
    return Status::OK();
  }

  /// \brief Get builder to append keys.
  ///
  /// Append a key with this builder should be followed by appending
  /// an item or null value with item_builder().
  ArrayBuilder* key_builder() const { return key_builder_.get(); }

  /// \brief Get builder to append items
  ///
  /// Appending an item with this builder should have been preceded
  /// by appending a key with key_builder().
  ArrayBuilder* item_builder() const { return item_builder_.get(); }

  /// \brief Get builder to add Map entries as struct values.
  ///
  /// This is used instead of key_builder()/item_builder() and allows
  /// the Map to be built as a list of struct values.
  ArrayBuilder* value_builder() const { return list_builder_->value_builder(); }

  std::shared_ptr<DataType> type() const override {
    // Key and Item builder may update types, but they don't contain the field names,
    // so we need to reconstruct the type. (See ARROW-13735.)
    return std::make_shared<MapType>(
        field(entries_name_,
              struct_({field(key_name_, key_builder_->type(), false),
                       field(item_name_, item_builder_->type(), item_nullable_)}),
              false),
        keys_sorted_);
  }

  Status ValidateOverflow(int64_t new_elements) {
    return list_builder_->ValidateOverflow(new_elements);
  }

 protected:
  inline Status AdjustStructBuilderLength();

 protected:
  bool keys_sorted_ = false;
  bool item_nullable_ = false;
  std::string entries_name_;
  std::string key_name_;
  std::string item_name_;
  std::shared_ptr<ListBuilder> list_builder_;
  std::shared_ptr<ArrayBuilder> key_builder_;
  std::shared_ptr<ArrayBuilder> item_builder_;
};

// ----------------------------------------------------------------------
// FixedSizeList builder

/// \class FixedSizeListBuilder
/// \brief Builder class for fixed-length list array value types
class ARROW_EXPORT FixedSizeListBuilder : public ArrayBuilder {
 public:
  using TypeClass = FixedSizeListType;

  /// Use this constructor to define the built array's type explicitly. If value_builder
  /// has indeterminate type, this builder will also.
  FixedSizeListBuilder(MemoryPool* pool,
                       std::shared_ptr<ArrayBuilder> const& value_builder,
                       int32_t list_size);

  /// Use this constructor to infer the built array's type. If value_builder has
  /// indeterminate type, this builder will also.
  FixedSizeListBuilder(MemoryPool* pool,
                       std::shared_ptr<ArrayBuilder> const& value_builder,
                       const std::shared_ptr<DataType>& type);

  Status Resize(int64_t capacity) override;
  void Reset() override;
  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<FixedSizeListArray>* out) { return FinishTyped(out); }

  /// \brief Append a valid fixed length list.
  ///
  /// This function affects only the validity bitmap; the child values must be appended
  /// using the child array builder.
  Status Append();

  /// \brief Vector append
  ///
  /// If passed, valid_bytes will be read and any zero byte
  /// will cause the corresponding slot to be null
  ///
  /// This function affects only the validity bitmap; the child values must be appended
  /// using the child array builder. This includes appending nulls for null lists.
  /// XXX this restriction is confusing, should this method be omitted?
  Status AppendValues(int64_t length, const uint8_t* valid_bytes = NULLPTR);

  /// \brief Append a null fixed length list.
  ///
  /// The child array builder will have the appropriate number of nulls appended
  /// automatically.
  Status AppendNull() final;

  /// \brief Append length null fixed length lists.
  ///
  /// The child array builder will have the appropriate number of nulls appended
  /// automatically.
  Status AppendNulls(int64_t length) final;

  Status ValidateOverflow(int64_t new_elements);

  Status AppendEmptyValue() final;

  Status AppendEmptyValues(int64_t length) final;

  Status AppendArraySlice(const ArraySpan& array, int64_t offset, int64_t length) final {
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    for (int64_t row = offset; row < offset + length; row++) {
      if (!validity || bit_util::GetBit(validity, array.offset + row)) {
        ARROW_RETURN_NOT_OK(value_builder_->AppendArraySlice(
            array.child_data[0], list_size_ * (array.offset + row), list_size_));
        ARROW_RETURN_NOT_OK(Append());
      } else {
        ARROW_RETURN_NOT_OK(AppendNull());
      }
    }
    return Status::OK();
  }

  ArrayBuilder* value_builder() const { return value_builder_.get(); }

  std::shared_ptr<DataType> type() const override {
    return fixed_size_list(value_field_->WithType(value_builder_->type()), list_size_);
  }

  // Cannot make this a static attribute because of linking issues
  static constexpr int64_t maximum_elements() {
    return std::numeric_limits<FixedSizeListType::offset_type>::max() - 1;
  }

 protected:
  std::shared_ptr<Field> value_field_;
  const int32_t list_size_;
  std::shared_ptr<ArrayBuilder> value_builder_;
};

// ----------------------------------------------------------------------
// Struct

// ---------------------------------------------------------------------------------
// StructArray builder
/// Append, Resize and Reserve methods are acting on StructBuilder.
/// Please make sure all these methods of all child-builders' are consistently
/// called to maintain data-structure consistency.
class ARROW_EXPORT StructBuilder : public ArrayBuilder {
 public:
  /// If any of field_builders has indeterminate type, this builder will also
  StructBuilder(const std::shared_ptr<DataType>& type, MemoryPool* pool,
             