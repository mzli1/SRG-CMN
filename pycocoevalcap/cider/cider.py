 const uint8_t* valid_bytes) final {
    ARROW_RETURN_NOT_OK(this->Reserve(length));
    this->UnsafeAppendToBitmap(valid_bytes, length);
    this->offsets_builder_.UnsafeAppend(offsets, length);
    this->sizes_builder_.UnsafeAppend(sizes, length);
    return Status::OK();
  }

  Status AppendValues(const offset_type* offsets, const offset_type* sizes,
                      int64_t length) {
    return AppendValues(offsets, sizes, length, /*valid_bytes=*/NULLPTR);
  }

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override {
    // Offset and sizes padding zeroed by BufferBuilder
    std::shared_ptr<Buffer> null_bitmap;
    std::shared_ptr<Buffer> offsets;
    std::shared_ptr<Buffer> sizes;
    ARROW_RETURN_NOT_OK(this->null_bitmap_builder_.Finish(&null_bitmap));
    ARROW_RETURN_NOT_OK(this->offsets_builder_.Finish(&offsets));
    ARROW_RETURN_NOT_OK(this->sizes_builder_.Finish(&sizes));

    if (this->value_builder_->length() == 0) {
      // Try to make sure we get a non-null values buffer (ARROW-2744)
      ARROW_RETURN_NOT_OK(this->value_builder_->Resize(0));
    }

    std::shared_ptr<ArrayData> items;
    ARROW_RETURN_NOT_OK(this->value_builder_->FinishInternal(&items));

    *out = ArrayData::Make(this->type(), this->length_,
                           {std::move(null_bitmap), std::move(offsets), std::move(sizes)},
                           {std::move(items)}, this->null_count_);
    this->Reset();
    return Status::OK();
  }

 protected:
  void UnsafeAppendEmptyDimensions(int64_t num_values) override {
    for (int64_t i = 0; i < num_values; ++i) {
      this->offsets_builder_.UnsafeAppend(0);
    }
    for (int64_t i = 0